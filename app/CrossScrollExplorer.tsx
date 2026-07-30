"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  RegionOverview,
  type FlakeAuditResult,
  type FlakeAuditSweep,
  type FlakeHoldoutResult,
  type FlakeResult,
  type Orbit,
  type RegionResult,
  type SheetletResult,
} from "./VolumeScene";

const SLICE_ORBIT: Orbit = { yaw: 0, pitch: 0, zoom: 1.72 };
const GRANULARITIES = [1, 2, 3, 4, 6, 8];
const DEFAULT_SLICE_INDEX = 3;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

function ordinal(value: number) {
  if (value === 2) return "2nd";
  if (value === 3) return "3rd";
  return `${value}th`;
}

export function CrossScrollExplorer() {
  const cacheRef = useRef(new Map<number, RegionResult>());
  const flakeCacheRef = useRef(new Map<number, FlakeResult>());
  const auditCacheRef = useRef(new Map<number, FlakeAuditResult>());
  const holdoutCacheRef = useRef(new Map<number, FlakeHoldoutResult>());
  const sheetletCacheRef = useRef(new Map<number, SheetletResult>());
  const [sliceIndex, setSliceIndex] = useState(DEFAULT_SLICE_INDEX);
  const [result, setResult] = useState<RegionResult | null>(null);
  const [granularity, setGranularity] = useState(2);
  const [glyphScale, setGlyphScale] = useState(0.35);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [message, setMessage] = useState("Loading the full-resolution field plane…");
  const [retry, setRetry] = useState(0);
  const [flakeResult, setFlakeResult] = useState<FlakeResult | null>(null);
  const [flakeState, setFlakeState] = useState<"loading" | "ready" | "error">("loading");
  const [flakeMessage, setFlakeMessage] = useState("Fitting local flake hypotheses…");
  const [auditResult, setAuditResult] = useState<FlakeAuditResult | null>(null);
  const [auditState, setAuditState] = useState<"loading" | "ready" | "error">("loading");
  const [auditMessage, setAuditMessage] = useState("Running rematched independence controls…");
  const [analysisSpacing, setAnalysisSpacing] = useState(64);
  const [holdoutResult, setHoldoutResult] = useState<FlakeHoldoutResult | null>(null);
  const [sheetletResult, setSheetletResult] = useState<SheetletResult | null>(null);
  const [validationState, setValidationState] = useState<"loading" | "ready" | "error">("loading");
  const [validationMessage, setValidationMessage] = useState("Cross-fitting disjoint needle partitions…");

  useEffect(() => {
    const cached = cacheRef.current.get(sliceIndex);
    if (cached) {
      setResult(cached);
      setState("ready");
      setMessage("");
      return;
    }
    const controller = new AbortController();
    setResult(null);
    setState("loading");
    setMessage("Loading the full-resolution field plane…");
    fetch(`/api/slab/overview?maxCells=100000&zIndex=${sliceIndex}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { error?: string } | null;
          throw new Error(payload?.error || `Field slice returned ${response.status}.`);
        }
        return response.json() as Promise<RegionResult>;
      })
      .then((payload) => {
        if (controller.signal.aborted) return;
        cacheRef.current.set(sliceIndex, payload);
        setResult(payload);
        setState("ready");
        setMessage("");
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setState("error");
        setMessage(error instanceof Error ? error.message : "The field slice is unavailable.");
      });
    return () => controller.abort();
  }, [retry, sliceIndex]);

  useEffect(() => {
    const cachedHoldout = holdoutCacheRef.current.get(sliceIndex);
    const cachedSheetlets = sheetletCacheRef.current.get(sliceIndex);
    if (cachedHoldout && cachedSheetlets) {
      setHoldoutResult(cachedHoldout);
      setSheetletResult(cachedSheetlets);
      setValidationState("ready");
      setValidationMessage("");
      return;
    }
    const controller = new AbortController();
    setHoldoutResult(cachedHoldout ?? null);
    setSheetletResult(cachedSheetlets ?? null);
    setValidationState("loading");
    setValidationMessage("Cross-fitting disjoint needle partitions and loading 3D sheetlets…");
    Promise.all([
      cachedHoldout
        ? Promise.resolve(cachedHoldout)
        : fetch(`/api/slab/flake-holdout?zIndex=${sliceIndex}&repetitions=4`, {
            signal: controller.signal,
          }).then(async (response) => {
            if (!response.ok) {
              const payload = (await response.json().catch(() => null)) as { error?: string } | null;
              throw new Error(payload?.error || `Held-out fit returned ${response.status}.`);
            }
            return response.json() as Promise<FlakeHoldoutResult>;
          }),
      cachedSheetlets
        ? Promise.resolve(cachedSheetlets)
        : fetch(`/api/slab/sheetlets?zIndex=${sliceIndex}&spacing=64`, {
            signal: controller.signal,
          }).then(async (response) => {
            if (!response.ok) {
              const payload = (await response.json().catch(() => null)) as { error?: string } | null;
              throw new Error(payload?.error || `Sheetlet graph returned ${response.status}.`);
            }
            return response.json() as Promise<SheetletResult>;
          }),
    ])
      .then(([holdout, sheetlets]) => {
        if (controller.signal.aborted) return;
        holdoutCacheRef.current.set(sliceIndex, holdout);
        sheetletCacheRef.current.set(sliceIndex, sheetlets);
        for (const cache of [holdoutCacheRef.current, sheetletCacheRef.current]) {
          while (cache.size > 2) {
            const oldest = cache.keys().next().value as number | undefined;
            if (oldest === undefined) break;
            cache.delete(oldest);
          }
        }
        setHoldoutResult(holdout);
        setSheetletResult(sheetlets);
        setValidationState("ready");
        setValidationMessage("");
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setValidationState("error");
        setValidationMessage(error instanceof Error ? error.message : "Held-out validation is unavailable.");
      });
    return () => controller.abort();
  }, [retry, sliceIndex]);

  useEffect(() => {
    const cached = auditCacheRef.current.get(sliceIndex);
    if (cached) {
      setAuditResult(cached);
      setAuditState("ready");
      setAuditMessage("");
      return;
    }
    const controller = new AbortController();
    setAuditResult(null);
    setAuditState("loading");
    setAuditMessage("Running rematched independence controls…");
    fetch(`/api/slab/flake-audit?zIndex=${sliceIndex}&repetitions=4`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { error?: string } | null;
          throw new Error(payload?.error || `Flake audit returned ${response.status}.`);
        }
        return response.json() as Promise<FlakeAuditResult>;
      })
      .then((payload) => {
        if (controller.signal.aborted) return;
        auditCacheRef.current.set(sliceIndex, payload);
        while (auditCacheRef.current.size > 2) {
          const oldest = auditCacheRef.current.keys().next().value as number | undefined;
          if (oldest === undefined) break;
          auditCacheRef.current.delete(oldest);
        }
        setAuditResult(payload);
        setAuditState("ready");
        setAuditMessage("");
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setAuditState("error");
        setAuditMessage(error instanceof Error ? error.message : "The independence audit is unavailable.");
      });
    return () => controller.abort();
  }, [retry, sliceIndex]);

  useEffect(() => {
    const cached = flakeCacheRef.current.get(sliceIndex);
    if (cached) {
      setFlakeResult(cached);
      setFlakeState("ready");
      setFlakeMessage("");
      return;
    }
    const controller = new AbortController();
    setFlakeResult(null);
    setFlakeState("loading");
    setFlakeMessage("Fitting local flake hypotheses…");
    fetch(`/api/slab/flakes?zIndex=${sliceIndex}&maxFlakes=3`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { error?: string } | null;
          throw new Error(payload?.error || `Flake plane returned ${response.status}.`);
        }
        return response.json() as Promise<FlakeResult>;
      })
      .then((payload) => {
        if (controller.signal.aborted) return;
        flakeCacheRef.current.set(sliceIndex, payload);
        while (flakeCacheRef.current.size > 2) {
          const oldest = flakeCacheRef.current.keys().next().value as number | undefined;
          if (oldest === undefined) break;
          flakeCacheRef.current.delete(oldest);
        }
        setFlakeResult(payload);
        setFlakeState("ready");
        setFlakeMessage("");
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setFlakeState("error");
        setFlakeMessage(error instanceof Error ? error.message : "Flake fitting is unavailable.");
      });
    return () => controller.abort();
  }, [retry, sliceIndex]);

  const availableZ = result?.grid.availableZ ?? [48, 80, 112, 144, 176, 208];
  const maximumSliceIndex = Math.max(availableZ.length - 1, 0);
  const localZ = availableZ[clamp(sliceIndex, 0, maximumSliceIndex)] ?? 0;
  const globalZ = (result?.globalOrigin.z ?? 7168) + localZ;
  const baseStep = result?.settings.gridStride ?? 32;
  const displayedCells = useMemo(() => {
    if (!result) return 0;
    return result.cells.reduce(
      (count, cell) =>
        count +
        (cell.valid && cell.index[0] % granularity === 0 && cell.index[1] % granularity === 0
          ? 1
          : 0),
      0,
    );
  }, [granularity, result]);
  const displayedFlakes = useMemo(() => {
    if (!flakeResult) return 0;
    return flakeResult.flakes.reduce(
      (count, flake) =>
        count +
        (flake.cellIndex[0] % granularity === 0 &&
        flake.cellIndex[1] % granularity === 0
          ? 1
          : 0),
      0,
    );
  }, [flakeResult, granularity]);
  const selectedAudit = useMemo<FlakeAuditSweep | null>(() => {
    return auditResult?.sweeps.find((sweep) => sweep.spacingVoxels === analysisSpacing) ?? null;
  }, [analysisSpacing, auditResult]);
  const displayedFlakeResult = useMemo<FlakeResult | null>(() => {
    if (!flakeResult) return null;
    const validation = new Map(
      holdoutResult?.validationByFlake.map((value) => [value.flakeId, value]) ?? [],
    );
    const sheetlets = new Map(
      sheetletResult?.nodes.map((value) => [value.flakeId, value]) ?? [],
    );
    return {
      ...flakeResult,
      flakes: flakeResult.flakes.map((flake) => {
        const heldout = validation.get(flake.id);
        const sheetlet = sheetlets.get(flake.id);
        return {
          ...flake,
          ...heldout,
          sheetletId: sheetlet?.sheetletId,
          sheetletSize: sheetlet?.sheetletSize,
          sheetletZSpanVoxels: sheetlet?.sheetletZSpanVoxels,
          sheetletDegree: sheetlet?.degree,
        };
      }),
      links: selectedAudit?.links ?? flakeResult.links,
      sheetletLinks: sheetletResult?.links,
    };
  }, [flakeResult, holdoutResult, selectedAudit, sheetletResult]);

  const selectSlice = (next: number) => {
    setSliceIndex(clamp(next, 0, maximumSliceIndex));
  };

  return (
    <main className="cross-scroll-page">
      <header className="cross-scroll-header">
        <nav className="cross-scroll-nav" aria-label="Experiment pages">
          <Link href="/" className="cross-scroll-back">
            ← Local workbench
          </Link>
          <Link href="/block-volume" className="cross-scroll-back">
            Solved block volume →
          </Link>
        </nav>
        <div>
          <p className="eyebrow">Acus · native transverse slab</p>
          <h1>Cross-scroll slice explorer</h1>
        </div>
        <p className="cross-scroll-summary">
          {state === "ready"
            ? flakeState === "ready"
              ? selectedAudit
                ? `${displayedFlakes.toLocaleString()} flakes · ${selectedAudit.observed.acceptedLinkCount.toLocaleString()} accepted ${analysisSpacing}-vox links · ${holdoutResult ? `${Math.round(holdoutResult.stats.validatedFullFlakeFraction * 100)}% held-out replicated` : validationMessage}`
                : `${displayedFlakes.toLocaleString()} flakes · ${auditMessage}`
              : `${displayedCells.toLocaleString()} Acus centers · ${flakeMessage}`
            : message}
        </p>
      </header>

      <section className="cross-scroll-toolbar" aria-label="Cross-scroll display controls">
        <div className="cross-scroll-slice-control">
          <div className="cross-scroll-control-heading">
            <span>Axial field plane</span>
            <strong>
              {sliceIndex + 1}/{maximumSliceIndex + 1} · local Z {localZ} · scan Z {globalZ}
            </strong>
          </div>
          <div className="cross-scroll-slider-row">
            <button
              type="button"
              className="region-metric-button"
              onClick={() => selectSlice(sliceIndex - 1)}
              disabled={sliceIndex <= 0 || state === "loading"}
              aria-label="Previous axial field plane"
            >
              −
            </button>
            <input
              type="range"
              min={0}
              max={maximumSliceIndex}
              step={1}
              value={sliceIndex}
              onChange={(event) => selectSlice(Number(event.target.value))}
              aria-label="Axial field plane"
            />
            <button
              type="button"
              className="region-metric-button"
              onClick={() => selectSlice(sliceIndex + 1)}
              disabled={sliceIndex >= maximumSliceIndex || state === "loading"}
              aria-label="Next axial field plane"
            >
              +
            </button>
          </div>
        </div>

        <label className="cross-scroll-select-control">
          <span>Analysis spacing</span>
          <select
            value={analysisSpacing}
            onChange={(event) => setAnalysisSpacing(Number(event.target.value))}
            aria-label="Independent flake link spacing"
          >
            <option value={32}>32 vox · 50% overlap</option>
            <option value={64}>64 vox · independent</option>
            <option value={96}>96 vox · 32-vox gap</option>
          </select>
        </label>

        <label className="cross-scroll-select-control">
          <span>Display granularity</span>
          <select value={granularity} onChange={(event) => setGranularity(Number(event.target.value))}>
            {GRANULARITIES.map((value) => (
              <option value={value} key={value}>
                {value === 1 ? "Every cell" : `Every ${ordinal(value)} cell`} · {baseStep * value} vox
              </option>
            ))}
          </select>
        </label>

        <label className="cross-scroll-scale-control">
          <span>Vector / flake scale</span>
          <strong>{glyphScale.toFixed(2)}×</strong>
          <input
            type="range"
            min={0.15}
            max={2.5}
            step={0.05}
            value={glyphScale}
            onChange={(event) => setGlyphScale(Number(event.target.value))}
          />
        </label>

        <div className="cross-scroll-audit-strip" aria-label="Rematched independence audit">
          {selectedAudit ? (
            <>
              <div>
                <span>Window evidence</span>
                <strong>
                  {selectedAudit.independentWindows
                    ? selectedAudit.gapVoxels
                      ? `${selectedAudit.gapVoxels}-vox gap`
                      : "non-overlapping"
                    : `${Math.round(selectedAudit.overlapFraction * 100)}% overlap`}
                </strong>
                <small>{Math.round((selectedAudit.observed.medianSharedNeedleFraction ?? 0) * 100)}% median shared needles</small>
              </div>
              <div>
                <span>Link survival</span>
                <strong>{Math.round(selectedAudit.linkSurvivalVs32 * 100)}%</strong>
                <small>{selectedAudit.observed.acceptedLinksPerCellPair.toFixed(3)} accepted links / cell pair</small>
              </div>
              <div>
                <span>Fiber · rematched</span>
                <strong>
                  {selectedAudit.observed.medianFiberAngleDeg?.toFixed(1) ?? "—"}° real ·{" "}
                  {selectedAudit.nulls.fiber.medianFiberAngleDeg?.toFixed(1) ?? "—"}° null
                </strong>
                <small>{selectedAudit.fiberNullDensityRatio.toFixed(1)}× null link density</small>
              </div>
              <div>
                <span>Depth phase</span>
                <strong>
                  {selectedAudit.observed.medianPositionResidualVoxels?.toFixed(1) ?? "—"} vox real ·{" "}
                  {selectedAudit.nulls.depth.medianPositionResidualVoxels?.toFixed(1) ?? "—"} null
                </strong>
                <small>{selectedAudit.nulls.depth.acceptedLinksPerCellPair.toFixed(3)} depth-null links / pair</small>
              </div>
              <div>
                <span>Spatial control</span>
                <strong>{selectedAudit.spatialNullDensityRatio.toFixed(1)}× null density</strong>
                <small>{selectedAudit.nulls.spatial.acceptedLinksPerCellPair.toFixed(4)} spatial-null links / pair</small>
              </div>
              <div>
                <span>Held-out replication</span>
                <strong>
                  {holdoutResult
                    ? `${Math.round(holdoutResult.stats.validatedFullFlakeFraction * 100)}% · ${holdoutResult.stats.validatedPairNullRatio.toFixed(1)}× null`
                    : validationState === "error"
                      ? "unavailable"
                      : "cross-fitting…"}
                </strong>
                <small>
                  {holdoutResult
                    ? `fold Δ ${holdoutResult.stats.medianFoldFiberDeltaDeg?.toFixed(1) ?? "—"}° / ${holdoutResult.stats.medianFoldDepthDeltaVoxels?.toFixed(1) ?? "—"} vox`
                    : validationMessage}
                </small>
              </div>
              <div>
                <span>3D sheetlet graph</span>
                <strong>
                  {sheetletResult
                    ? `${sheetletResult.stats.sheetletCount.toLocaleString()} · ${sheetletResult.stats.linkNullRatio.toFixed(1)}× null`
                    : validationState === "error"
                      ? "unavailable"
                      : "assembling…"}
                </strong>
                <small>
                  {sheetletResult
                    ? `${sheetletResult.stats.acceptedZLinkCount.toLocaleString()} Z links · largest ${sheetletResult.stats.largestSheetletSize}`
                    : validationMessage}
                </small>
              </div>
            </>
          ) : (
            <p data-state={auditState}>{auditMessage}</p>
          )}
        </div>
      </section>

      <section className="cross-scroll-viewer" aria-live="polite">
        {result ? (
          <RegionOverview
            key={sliceIndex}
            result={result}
            title={`Axial Acus field · scan Z ${globalZ}`}
            defaultVectorMode="sheetlets"
            defaultOrbit={SLICE_ORBIT}
            granularity={granularity}
            glyphScale={glyphScale}
            flakeResult={displayedFlakeResult}
            flakeLoading={flakeState === "loading"}
            auditSweep={selectedAudit}
          />
        ) : (
          <div className="cross-scroll-loading" data-state={state}>
            <strong>{state === "error" ? "Field slice unavailable" : "Loading field slice"}</strong>
            <p>{message}</p>
            {state === "error" ? (
              <button type="button" className="region-metric-button" onClick={() => setRetry((value) => value + 1)}>
                Retry
              </button>
            ) : null}
          </div>
        )}
      </section>
    </main>
  );
}
