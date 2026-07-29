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
   - Relative normal signs may be synchronized for ordering, but each connected
     region retains an arbitrary sign. Inward/outward and recto/verso are later
     physical interpretations.
   - Neighboring flake sequences use order-preserving partial matching.
     Missing modes are explicit births/deaths; equal ordinal numbers are not
     required, and crossings are forbidden.
   - Collision-safe graph components are local surface branches only.

4. **Sparse association**
   - Future branch joins should use compatible facing edges, fiber continuation,
     order feasibility, and material accountability.
   - A completeness prior applies only to flakes with a calibrated reliability
     estimate. Every solve retains an explicit outlier/unassigned state.
   - Secondary-normal fragments remain separate branches until the association
     model can represent family ambiguity rather than absorbing them as extreme
     curvature.

5. **Physical interpretation**
   - Page identity, winding order, side, and recto/verso are downstream
     interpretations. None is encoded in voxel labels, flake IDs, or branch
     component IDs.

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

The first monotone prototype uses the densest 32 x 32 x 14-cell primary-family
window. Relative signs form one connected region with 0.83% weighted sign
frustration. Compared with the active graph it:

- preserves 49,549 of 49,745 retained links;
- removes all 149 pairwise order crossings;
- adds 112 compatible non-crossing links;
- leaves median edge residual and fiber disagreement unchanged at about 0.73
  voxels and 1.08 degrees; and
- changes 3,361 local branches into 3,349, with the largest branch changing
  from 372 to 361 flakes.

This is a useful negative result: the current pairwise geometry is already
almost order-consistent, so ordinal mistakes are not the main source of
fragmentation. The next useful solve is branch-level sparse association across
missing continuation edges, with order feasibility and material accountability
as constraints. It should not be another local ordinal rematcher or a dense
global surface fit.

## Reproducible commands

```bash
.venv/bin/python scripts/science-ci.py \
  --root work/cross-scroll-analysis-z512 --verify-artifacts

.venv/bin/python scripts/build-material-intervals.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/prototype-monotone-layers.py \
  --root work/cross-scroll-analysis-z512
```

The large NumPy products remain ignored under `work/`. Their summaries contain
content hashes for all generated arrays and input identities. The compact
science baseline is committed under `benchmarks/`.
