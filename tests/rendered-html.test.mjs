import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${path}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the local rectifier workbench", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Rectifier Lab \| Local voxel workbench<\/title>/i);
  assert.match(html, /Rectifier Lab/);
  assert.match(html, /Compare local fiber geometry across the loaded scan\./);
  assert.match(html, /Axial/);
  assert.match(html, /Coronal/);
  assert.match(html, /Sagittal/);
  assert.match(html, /Local Acus inspector/);
  assert.match(html, /Fit surface at seed/);
  assert.match(html, /Rectified sample/);
  assert.match(html, /role="status"/);
  assert.doesNotMatch(html, /Your site is taking shape|codex-preview|SkeletonPreview/);
});

test("server-renders the dedicated cross-scroll explorer", async () => {
  const response = await render("/cross-scroll");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Cross-scroll slice explorer/);
  assert.match(html, /Display granularity/);
  assert.match(html, /Analysis spacing/);
  assert.match(html, /Vector \/ flake scale/);
  assert.match(html, /Loading field slice/);
  assert.match(html, /Running rematched independence controls/);
});

test("keeps the API contract and coordinate mapping explicit", async () => {
  const [workbench, volumeScene, explorer, page, layout, css] = await Promise.all([
    readFile(new URL("../app/RectifierWorkbench.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/VolumeScene.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/CrossScrollExplorer.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(workbench, /window\.location\.origin/);
  assert.match(workbench, /\/api\/volume/);
  assert.match(workbench, /\/api\/slice\?/);
  assert.match(workbench, /\/api\/fit/);
  assert.match(workbench, /\/health/);
  assert.match(workbench, /body: JSON\.stringify\(\{ seed, \.\.\.parameters \}\)/);

  assert.match(
    workbench,
    /axis: "z", title: "Axial", code: "XY · Z", horizontal: "x", vertical: "y"/,
  );
  assert.match(
    workbench,
    /axis: "y", title: "Coronal", code: "XZ · Y", horizontal: "x", vertical: "z"/,
  );
  assert.match(
    workbench,
    /axis: "x", title: "Sagittal", code: "YZ · X", horizontal: "y", vertical: "z"/,
  );
  assert.match(workbench, /aria-label="Orbitable 3D preview/);
  assert.match(workbench, /addEventListener\("wheel", handleWheel, \{ passive: false \}\)/);
  assert.match(workbench, /pinch zoom · two-finger pan/);
  assert.match(workbench, /, 1, 32\)/);
  assert.match(workbench, /, 0\.35, 16\)/);
  assert.match(workbench, /type="range"/);
  assert.match(workbench, /type="number"/);
  assert.match(volumeScene, /\/api\/cube\?/);
  assert.match(volumeScene, /\/api\/needles/);
  assert.match(volumeScene, /\/api\/field/);
  assert.match(volumeScene, /\/api\/audit/);
  assert.match(volumeScene, /\/api\/padding-audit/);
  assert.match(volumeScene, /\/api\/region/);
  assert.match(volumeScene, /Fit Acus/);
  assert.match(volumeScene, /Map 3×3/);
  assert.match(volumeScene, /Normal deviation/);
  assert.match(volumeScene, /Profile match/);
  assert.match(volumeScene, /Best depth lag/);
  assert.match(volumeScene, /Audit 4–32/);
  assert.match(volumeScene, /Audit halo/);
  assert.match(volumeScene, /Analyze current crop/);
  assert.match(volumeScene, /Cross-scroll slab analysis progress/);
  assert.match(volumeScene, /Open full-window slice explorer/);
  assert.match(volumeScene, /XY page tangent t/);
  assert.match(volumeScene, /rotate90\(projectXY\(n\)\)/);
  assert.match(explorer, /defaultVectorMode="sheetlets"/);
  assert.match(explorer, /maxCells=100000&zIndex=/);
  assert.match(explorer, /\/api\/slab\/flakes\?zIndex=/);
  assert.match(explorer, /\/api\/slab\/flake-audit\?zIndex=/);
  assert.match(explorer, /\/api\/slab\/flake-holdout\?zIndex=/);
  assert.match(explorer, /\/api\/slab\/sheetlets\?zIndex=/);
  assert.match(explorer, /Display granularity/);
  assert.match(explorer, /Analysis spacing/);
  assert.match(explorer, /64 vox · independent/);
  assert.match(explorer, /Fiber · rematched/);
  assert.match(explorer, /Spatial control/);
  assert.match(explorer, /Held-out replication/);
  assert.match(explorer, /3D sheetlet graph/);
  assert.match(explorer, /Vector \/ flake scale/);
  assert.match(volumeScene, /Fiber flakes/);
  assert.match(volumeScene, /Linked tracks/);
  assert.match(volumeScene, /3D sheetlets/);
  assert.match(volumeScene, /Held-out fit/);
  assert.match(volumeScene, /Depth phase/);
  assert.match(volumeScene, /sharedNeedleFraction/);
  assert.match(volumeScene, /acus-compute-key/);
  assert.match(volumeScene, /computeBackend\.toUpperCase/);
  assert.match(volumeScene, /Finite support length/);
  assert.match(volumeScene, /Real scan-data halo/);
  assert.match(volumeScene, /setContextPadding\(nextLength\)/);
  assert.match(volumeScene, /COHERENCE VS CENTER SPACING/);
  assert.match(volumeScene, /shuffled null/);
  assert.match(volumeScene, /shared normal/);
  assert.match(volumeScene, /ORIENTATION ALONG n/);
  assert.match(volumeScene, /signed position along n/);
  assert.match(volumeScene, /meanTwoModeCoverage/);
  assert.match(volumeScene, /unsigned 0° ≡ 180°/);
  assert.match(volumeScene, /className="volume-overlay"/);
  assert.match(volumeScene, /sampler3D/);
  assert.match(volumeScene, /gl\.TEXTURE_3D/);
  assert.match(volumeScene, /cubeSize \*\* 3/);
  assert.match(volumeScene, /centered at the selected seed/);

  assert.match(page, /<RectifierWorkbench \/>/);
  assert.match(layout, /Rectifier Lab \| Local voxel workbench/);
  assert.doesNotMatch(layout, /Starter Project|next\/font/);
  assert.match(css, /@media \(max-width: 720px\)/);
  assert.match(css, /canvas:focus-visible/);
  assert.match(css, /\.volume-overlay/);
  assert.match(css, /\.acus-legend/);
  assert.match(css, /\.orientation-profile/);
  assert.match(css, /\.acus-field-panel/);
  assert.match(css, /\.field-cell-grid/);
  assert.match(css, /\.acus-audit-panel/);
  assert.match(css, /\.audit-table/);
  assert.match(css, /\.padding-audit-panel/);
  assert.match(css, /\.padding-audit-button/);
  assert.match(css, /\.region-overview/);
  assert.match(css, /\.region-bake-button/);
});
