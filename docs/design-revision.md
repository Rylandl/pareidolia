# Evidence-to-layer design revision

This revision separates observations, local hypotheses, sparse associations,
and physical interpretation. The pipeline may build globally consistent sparse
graphs, but it does not perform a global dense-geometry solve and does not call
a graph component a physical papyrus sheet.

## Layered contracts

1. **Native evidence**
   - CT intensity and finite-support Acus needles are observations.
   - Normal families are local, unsigned orientation hypotheses with disjoint
     needle ownership.
   - Replication measurements estimate the probability that a local flake is
     rediscovered. They are not probabilities of physical sheet identity.

2. **Hypothesis-independent material accountability**
   - Every accepted local normal has a native-CT profile from -32 through +32
     voxels at one-voxel spacing.
   - The declared analysis air cutoff produces a material mask before any flake or
     carrier is consulted.
   - A separate claim overlay labels samples as air, unassigned material,
     singly claimed material, or contested material.
   - Contested means that multiple separated local flake-depth clusters occupy
     one unresolved CT material run. It is an ambiguity/contact diagnostic,
     not evidence of two particular sheets.
   - Apparent CT thickness is permitted only for singly claimed intervals that
     do not touch either profile boundary. Fitted flake thickness is never
     substituted for physical papyrus thickness.

3. **Local surface branches**
   - Flakes are linked by finite-patch edge agreement and transported fiber
     direction. A changing normal is curvature, not an identity failure.
   - Normal and fiber directions remain unsigned axes. For each adjacent cell,
     both relative depth orientations are solved independently. The higher-value
     order-preserving partial match wins; an exact score tie retains only links
     present in both orientations.
   - A retained link stores relative parity between two raw depth coordinates,
     not an absolute side. Parity transforms with either input coordinate and
     therefore does not encode inward/outward or recto/verso.
   - Missing modes are explicit births/deaths. Collision-safe links are processed
     in descending score, and a link that closes a parity-inconsistent cycle is
     explicitly deferred.
   - Collision-safe graph components are local surface branches only.

4. **Sparse association**
   - Branch joins use compatible facing edges, fiber continuation, order
     feasibility, and material accountability.
   - A join must be accepted in the main chunk and every overlapping subwindow
     that observes it. It is then rebuilt with the active MLS carrier; this is a
     same-sample construction gate, not independent validation.
   - Pairwise-incoherent joins are deferred. If individually coherent joins make
     an incoherent transitive association, the weakest retained construction
     edge is removed until every output association passes the carrier gate.
   - Branch-relative gauges are aligned only through parity votes in cells two
     branches actually share. Tied or frustrated parity observations do not
     become hard order edges.
   - Explicit window origins produce separate, hash-identified artifacts.
     Reconciliation uses stable source-flake identities in the geometric
     overlap: raw matches, collision-pruned edges, and branch joins are compared
     separately, and any one-window-only decision remains deferred.
   - A completeness prior applies only to flakes with a calibrated reliability
     estimate. Every solve retains an explicit outlier/unassigned state.
   - Secondary-normal fragments remain separate branches until the association
     model can represent family ambiguity rather than absorbing them as extreme
     curvature.

5. **Physical interpretation**
   - Page identity, winding order, side, and recto/verso are downstream
   interpretations. None is encoded in voxel labels, flake IDs, or branch
   component IDs.

6. **Cross-association integrity**
   - Every accepted merged association is reconstructed as the active MLS
     carrier and triangulated for an explicit non-intersection audit.
   - Intersections within 24 voxels of supporting flakes on both surfaces are
     separated from intersections in the wider carrier support skirt.
   - A bidirectional sampled-clearance sweep is reported descriptively. It is
     not called papyrus thickness, and shared-cell depth order distinguishes
     an ordered near-contact from an unexplained boundary approach.
   - Any local association involved in a support-skirt or evidence-core mesh
     intersection is excluded from the consensus join graph and preserved in a
     separate quarantine catalog.

## Current z512 findings

The active multi-normal benchmark freezes 37 artifacts (781,614,775 bytes) and
nine preservation/construction guards. It deliberately excludes stale
single-normal carrier descendants.

The full material census samples 261,302 profiles and 16,984,630 native CT
positions in 26 seconds. At the declared analysis air threshold:

- 96.32% of positions are material;
- 243,069 profiles contain material across the entire 65-voxel window;
- 234,496 profiles contain contested material;
- 695,183 of 724,672 local flake claims are supported by material; and
- only 72 intervals qualify for the deliberately narrow apparent-thickness
  statistic.

A deterministic 20,000-profile threshold sweep confirms that higher arbitrary
cutoffs create apparent separators by discarding evidence. At raw threshold 96,
for example, claims are separated across CT runs in 50.87% of profiles, but
claim support has already fallen to 78.27%. Air/material thresholding is
therefore useful for accountability and outlier handling, not as the layer
solver.

The original window-wide sign synchronization was removed after a full overlap
census showed that crop-dependent spanning trees could orient different parts
of the same unsigned line field differently. Those disagreements were
algorithmic gauge instability, not physical normal evidence. The replacement
axial matcher is invariant to flipping either cell's normal/depth coordinate.

In a representative 32 x 32 x 14-cell dense window, the axial solve produces
47,515 raw non-crossing links. Collision pruning retains 45,535 and explicit
parity-cycle pruning removes another 238, leaving 45,297 links and 4,005 linked
branches. A neighboring independent solve shares 10,495 flakes and 11,476 raw
links. Every stable identity, raw depth, raw link, and relative parity agrees
exactly. Only the later collision/parity graph solve is crop-dependent: 70
retained links differ, for 99.36% retained-edge agreement.

The complete 242 x 242 x 14-cell census uses 51 occupied 32 x 32 x 14 windows,
24-cell XY strides, and 86 face-overlap reconciliations. Forty-nine possible
windows contain no primary claim and are skipped. A fresh four-worker pass
takes 310 seconds and visits 1,251,591 window-local flake observations while
covering all 704,145 unique primary flakes. Its central result is exact raw
invariance:

- all 818,414 unique raw matches are accepted by every observing window;
- all 435,661 multiply observed raw matches have unanimous relative parity;
- repeated edge scores have a zero range to floating-point precision;
- collision/parity pruning leaves 2,148 context-dependent retained edges out of
  785,595 unique retained edges; and
- no absolute normal sign or side exists in the schema.

The local branch-association passes retain 650 exact-coherent join occurrences
and defer 542 pair candidates at the exact MLS gate. Cross-association integrity
finds seven intersecting local carrier pairs, five within both evidence cores.
Those pairs implicate 14 local associations and 14 accepted join occurrences.
The integrity quarantine removes 12 unique endpoint pairs from consensus while
leaving them fully traceable. Of the remaining joins, 125 are observed and
accepted in at least two windows; single-window joins are not called
overlap-validated.

The unanimous raw catalog supports one whole-volume sparse graph solve without
claiming a global surface. It completes in 12.1 seconds. Starting from 818,414
edges, descending-score cell-collision pruning rejects 29,151 and parity-cycle
consistency rejects 5,417 more. The final 783,846-edge graph has:

- 704,145 nodes, of which 636,717 are linked;
- 63,783 linked collision-free branches;
- zero within-component cell collisions;
- a largest branch of 586 flakes;
- 531 branches spanning at least 11 of 14 axial planes; and
- 235 branches spanning all 14 planes.

This is the first region-wide sparse construction result. Components remain
surface branches rather than sheets; the next association stage may consume
only overlap-validated, integrity-clean gap joins and must repeat collision and
exact-geometry checks after any transitive merge.

## Reproducible commands

```bash
.venv/bin/python scripts/science-ci.py \
  --root work/cross-scroll-analysis-z512 --verify-artifacts

.venv/bin/python scripts/build-material-intervals.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/prototype-monotone-layers.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/associate-monotone-branches.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/audit-branch-association-integrity.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/prototype-monotone-layers.py \
  --root work/cross-scroll-analysis-z512 \
  --window-origin-cell-xyz 141 74 0

.venv/bin/python scripts/associate-monotone-branches.py \
  --root work/cross-scroll-analysis-z512 \
  --window-origin-cell-xyz 141 74 0

.venv/bin/python scripts/reconcile-overlapping-windows.py \
  --root work/cross-scroll-analysis-z512 \
  --target-window-origin-cell-xyz 141 74 0

.venv/bin/python scripts/run-window-schedule.py \
  --root work/cross-scroll-analysis-z512 --maximum-workers 4

.venv/bin/python scripts/build-global-monotone-graph.py \
  --root work/cross-scroll-analysis-z512
```

The large NumPy products remain ignored under `work/`. Their summaries contain
content hashes for all generated arrays and input identities. The compact
science baseline is committed under `benchmarks/`.
