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
   - A clean local join must be accepted in the main chunk and every overlapping
     subwindow that observes it. It is then rebuilt with the active MLS carrier;
     this is a same-sample construction gate, not independent validation.
   - Pairwise-incoherent joins are deferred. If individually coherent joins make
     an incoherent transitive association, the weakest retained construction
     edge is removed until every output association passes the carrier gate.
   - Branch-relative gauges are aligned only through parity votes in cells two
     branches actually share. Tied or frustrated parity observations do not
     become hard order edges.
   - Explicit window origins produce separate, hash-identified artifacts.
     Reconciliation uses stable source-flake identities in the geometric
     overlap: raw matches, collision-pruned edges, and branch joins are compared
     separately. Window disagreement is retained as weaker provenance rather
     than relabeled as independent validation.
   - The whole-volume association solve keeps overlap-validated, unanimous
     single-window, context-disputed, subwindow-unresolved, and locally exact-
     deferred evidence in separate tiers. The last two enter through an explicit
     candidate catalog only after passing local score, material, order, and
     collision checks. Every novel endpoint pair must still pass complete-global-
     branch reconstruction, collision safety, transitive reconstruction, and
     mesh integrity. Local integrity quarantine remains a hard exclusion.
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
   - Any intersecting local association is excluded from clean consensus and
     preserved in a separate quarantine catalog. In the whole-volume solve, an
     intersecting carrier pair loses only its weakest retained construction edge
     before reconstruction and re-audit; evidence provenance breaks exact score
     ties. Iteration stops only at a zero-intersection fixed point.

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

The accepted endpoint catalog preserves three provenance tiers. Tier one is the
125 integrity-clean joins accepted by every one of at least two observing
windows. Tier two contains 214 unanimous single-window candidates. Tier three
contains 172 endpoint pairs accepted in only a subset of their observing
windows. This third tier is explicitly named `context-disputed`; it is not called
validation. Two disputed pairs remain excluded by local mesh integrity and 13
already lie inside one global branch. The accepted-evidence scope therefore has
496 novel candidates: 125 overlap-validated, 214 single-window, and 157 context-
disputed. Its 479 retained joins exactly reproduce the v3 result.

Candidate discovery then inspects local evidence withheld before final
association rather than jumping directly to an unrestricted spatial search. The
51 windows contain 47,981 candidate occurrences over 29,809 unique endpoint
pairs. The rescue catalog admits only two cases: a pair accepted by the main
window but not unanimously by its subwindows, or a subwindow-stable pair that
failed the small-window MLS fit. Material-deferred, order-ambiguous, order-
blocked, below-score, collision-rejected, already accepted, locally quarantined,
already-linked, and duplicate global branch pairs remain excluded.

After mapping endpoint identities into the whole-volume graph, the catalog has
230 subwindow-unresolved and 429 locally exact-deferred global branch pairs. It
retains all 1,045 supporting local observations, including scores, overlap
counts, original decisions, and available local residuals. Of the 230 unresolved
pairs, 229 have insufficient subwindow observations and one has an observed
disagreement. This artifact is a candidate inventory only; it accepts no
association.

Every candidate is then reconstructed from its complete global branches. The
accepted tiers pass 483 of 496 pair gates. Global context also passes 111 of 230
subwindow-unresolved pairs and five of 429 locally exact-deferred pairs. The five
rescues are narrow rather than permissive: their local failures sit near the
normal/height gates, while complete-branch fits range from 0.67 to 2.47 voxels
and 3.06 to 5.67 degrees. Across all 1,155 candidates, 599 pass complete pair
geometry; 494 are deferred with an incoherent input carrier and 62 fail despite
coherent inputs.

Descending-score construction retains 593 of those pairwise passes initially;
six would repeat an Acus cell. All 562 resulting transitive carriers pass exact
reconstruction without pruning. The first mesh audit finds eight intersecting
carrier pairs, all in the support skirt and none in both evidence cores. Removing
the six weakest implicated construction edges—three context-disputed and three
subwindow-unresolved—produces a zero-intersection fixed point.

The final 587 joins form 556 associations over 1,143 branches and 12,042 flakes:
528 pairs, 25 triples, and three four-branch groups. There are 117 overlap-only,
202 single-window-only, 132 context-disputed-only, 95 subwindow-unresolved-only,
four local-exact-rescue-only, and six mixed-provenance associations. Final
carrier medians are 1.25 voxels in height and 3.90 degrees in normal, with maxima
of 2.94 voxels and 5.99 degrees. Triangulating all 556 carriers produces 524
overlapping bounding-box pairs, 127,776 broad-phase triangle pairs, and 3,802
narrow-phase checks with zero final support-skirt or evidence-core intersections.
Twenty-five associations span at least 11 axial planes and nine span all 14.

At catalog scale, association reduces 131,211 original components to 130,624
without changing the 586-flake maximum. Relative to the original sparse graph,
the final catalog has 25 more fragments with at least 25 flakes, 11 more with at
least 50, four more with at least 100, nine more spanning at least 11 planes, and
six more spanning all 14. Relative to v3, the rescue tiers add eight, two, zero,
one, and zero to those respective counts. Of the 556 merged associations, 320
extend axial span and 42 add at least ten flakes beyond their largest input
branch.

Fresh candidate discovery takes 11.2 seconds and the five-tier solve 34.5
seconds, reducing 63,783 linked branches to 63,196 branch groups. An
`--accepted-only` compatibility solve exactly reproduces all 23 compared
geometric and decision arrays from v3, including association identities;
`--clean-only` retains the earlier two-tier scope. This remains a deliberately
narrow region-wide construction result, not a sheet census. Unassociated global
branches are outside the carrier-intersection audit, and no provenance tier is
relabeled as independent replication. Output identities remain sparse exact-
coherent surface hypotheses rather than pages or physical papyrus layers.

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

.venv/bin/python scripts/build-global-branch-candidates.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/associate-global-branches.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/associate-global-branches.py \
  --root work/cross-scroll-analysis-z512 --accepted-only

.venv/bin/python scripts/associate-global-branches.py \
  --root work/cross-scroll-analysis-z512 --clean-only
```

The large NumPy products remain ignored under `work/`. Their summaries contain
content hashes for all generated arrays and input identities. The compact
science baseline is committed under `benchmarks/`.
