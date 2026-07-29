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
     single-window, context-disputed, subwindow-unresolved, locally exact-
     deferred, directional-boundary, and locally order-resolved boundary
     evidence in separate tiers. The first five originate in local windows.
     Directional-boundary discovery is a low-priority construction heuristic
     for exposed degree-one through degree-six graph nodes and cannot displace
     any local-evidence join. A whole-volume cyclic-order deferral enters one
     still-lower tier only when at least two tiled windows observe both
     endpoints and every observation is locally acyclic and unblocked.
     Every novel pair must still pass complete-global-branch reconstruction,
     collision safety, transitive reconstruction, and mesh integrity. Local
     integrity quarantine remains a hard exclusion.
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
`--clean-only` retains the earlier two-tier scope.

The next audit asks whether local windows fail to propose plausible fragment
boundaries. Repeating the original degree-one endpoint search over the complete
global graph finds 27,446 scored branch pairs; 27,445 were already present in a
local candidate list, and the only novel pair is below threshold. Window edges
are therefore not the bottleneck under the old endpoint definition.

The broader boundary model includes degree-one through degree-six nodes whose
retained tangent-neighbor directions have a resultant concentration of at least
0.25. The open direction is the opposite resultant, and no retained neighbor may
already occupy that cone with cosine above 0.50. This is an edge-exposure
heuristic, not a physical claim about papyrus boundaries. Of 636,717 eligible
nodes, 426,766 are directionally exposed. A streamed vectorized search evaluates
106,943,713 nearby node pairs, 15,356,504 of which mutually face; the active
geometry model leaves 249,135 hits over 97,773 branch pairs. After removing every
branch pair ever proposed locally, 68,899 scored pairs remain. Score, material,
global order, and collision gates select only 276. Candidate discovery takes
30.0 seconds. The bounded local-order audit below brings total construction time
to 37.7 seconds and stores every decision, observation, and geometric diagnostic
in `global-boundary-candidates-v4`.

All local-evidence joins are constructed before this new tier, and exact or
integrity pruning always sacrifices a directional-only edge before any local
edge. The `--local-evidence-only` compatibility solve consequently matches all
60 common v4 artifact arrays exactly. Complete-branch gates defer 139 of the 276
because an input branch is not a coherent carrier and six more at pair geometry.
One candidate creates a cell collision, one is removed by transitive exact
reconstruction, and five are removed by cross-carrier integrity. The remaining
124 directional joins are strictly additive to the 587 v4 joins.

The v5 result has 711 joins in 672 associations over 1,383 branches and 14,509
flakes: 637 pairs, 31 triples, and four four-branch groups. Final median carrier
residuals are 1.24 voxels and 3.90 degrees, with maxima of 2.94 voxels and 5.99
degrees. The final audit processes 699 overlapping carrier boxes, 172,091 broad-
phase triangle pairs, and 4,086 narrow-phase pairs with zero intersections. The
120 associations containing a directional join extend axial span in 74 cases;
21 add at least ten flakes beyond their largest v4 input and one adds at least
25. One hundred two of the 124 retained joins have both endpoints in contested
material, so these remain geometry-conditioned constructions rather than
replication evidence.

At full-catalog scale v5 reduces 131,211 original components to 130,500 without
changing the 586-flake maximum. Relative to v4 it adds 12 components at least 25
flakes, eight at least 50, none at least 100, six spanning at least 11 planes,
and three spanning all 14. The count at least ten decreases by one because two
already-qualifying fragments merge. This is measurable gap closure without a
giant transitive collapse.

The whole-volume order graph is not a useful hard veto everywhere: one cyclic
strongly connected component contains 54,622 branches, and 5,425 otherwise
scored novel boundaries are deferred as order-ambiguous. Every one is observed
in at least one tiled window; 2,707 are observed in two or four. Rebuilding the
order condensation independently inside each observing window gives 9,064
bounded observations: 1,701 feasible, 70 already on one local branch, 1,190
locally ordered apart, and 6,103 still cyclic. Recovery requires at least two
observations and unanimity among all observing windows. It therefore admits 307
candidates, while 441 clean single-window cases and every blocked or cyclic
case remain deferred.

These 307 candidates are constructed after the v5 evidence and are the first
edges removed by either exact or integrity pruning. Complete-branch gates defer
122 because an input branch is not a coherent carrier and 15 more at pair
geometry; five collide, one fails transitive exact reconstruction, and 11 are
removed by cross-carrier integrity. The remaining 153 joins form 150 affected
associations, including three with two recovered joins. All 1,431 pre-existing
candidate decisions and exact diagnostics are identical after stable endpoint
alignment, and the v5 partition is never split. The v6 result has 864 retained
joins in 812 associations over 1,676 branches and 17,812 flakes. Final median
carrier residuals are 1.24 voxels and 3.92 degrees, with maxima of 2.98 voxels
and 6.00 degrees and zero intersections after 372,118 broad-phase and 9,439
narrow-phase triangle checks.

Of the 150 affected associations, 82 gain axial span and 18 add at least ten
flakes beyond their largest v5 input. At full-catalog scale v6 reduces the
component count from 130,500 to 130,347, adds eight fragments with at least 25
flakes, five with at least 50, one with at least 100, two spanning at least 11
planes, and one spanning all 14; the 586-flake maximum is unchanged. One hundred
fifty-two of the 153 retained joins have both endpoints in contested material,
so the result remains geometry-conditioned gap closure rather than independent
evidence for a physical sheet.

## Fragment-termination census and targeted Acus queue

The next stage diagnoses why substantial v6 fragments stop before changing any
association rule. The census is deliberately restricted to degree-one nodes in
the global monotone graph whose final association contains at least 25 flakes.
It reconciles all 47,981 local candidate occurrences with the novel global
boundary candidates and the final complete-branch decisions. The category on a
termination records the furthest stage reached by current evidence: no
compatible candidate, weak geometry, material deferral, local-order failure,
overlap instability, exact geometry failure, collision, integrity quarantine,
or an accepted continuation. This is a pipeline diagnosis, not a claim that a
termination is a physical papyrus edge.

There are 32,112 definite ends in scope. Only 319 are endpoints of a continuation
already accepted by v6, leaving 31,793 unresolved. Of these, 31,374 have a
stable outward tangent. Grouping adjacent cells only within one final
association and only when outward tangents have cosine at least 0.50 produces
26,078 termination regions; 419 ends without a usable tangent remain in the
endpoint artifact but are not assigned a spatial target. The cluster-level
breakdown is:

- 13,294 with no compatible candidate, 10,954 with only weak geometry, and two
  with unresolved overlap support;
- 1,471 order-ambiguous and 131 order-blocked;
- 97 rejected by complete-branch geometry, five by collision, and six by the
  integrity quarantine; and
- 118 deferred by the material model.

The follow-up queue projects each evidence-poor region 32 voxels along its
outward tangent and samples a 32-cubed CT neighborhood. Among the 512
highest-priority samples, 507 are nontruncated and exceed the deliberately loose
0.35 material-fraction gate; their median material fraction is 0.9998. A cap of
two targets per association then selects 128 targets across 68 associations.
Every selected target is in the weak-geometry class: these have an observed
nearby continuation that failed the present geometric path and are therefore a
higher-yield dense-Acus experiment than order or integrity failures. The latter
remain separate ranked review queues rather than being silently converted into
reanalysis requests.

The initial cold census takes 32.2 seconds and a forced rerun with the filesystem
cache warm takes 6.9 seconds. Exact endpoint classifications, cluster geometry,
target coordinates, CT metrics, priorities, and queue membership are stored in
`fragment-termination-census-v1.npz`; the compact contract, counts, and ranked
queues are in `fragment-termination-census-v1.json`.

## Targeted dense Acus at open fragment termini

The termination queue is evaluated by comparing two fits at the same spatial
location. The control uses the existing whole-volume catalog: 4-voxel candidate
spacing, at most 32 needles per 32-cubed catalog bin, and at most 160 needles in
one 64-cubed fitting neighborhood. The dense path rereads the native CT with
2-voxel candidate spacing, permits 256 candidates per catalog-sized bin and
1,024 needles per target, and fits up to 12 depth/fiber modes. Needle length,
halo, Hessian scale, normalization, fit geometry, and downstream score gates
remain unchanged. Candidate blocks and 32-voxel retention bins are anchored to
the global scan coordinates, so a target gives exactly the same result alone or
when coalesced with a neighboring GPU crop.

Endpoint resolution removes an important false target class before extraction.
Forty of the 128 queued clusters have every one-cell outward member target
already occupied by their own final v6 association. They are internal
degree-one branch ends, not open association boundaries, and remain recorded as
`association-covered-next-cell`. The other 88 targets are unique open
association cells and coalesce into 81 crops of at most 3,932,160 voxels.

Both catalogs are fit at each target's identical 64-cube center and scored
against the same local association context. A pass requires a construction
score of at least 0.55, a best-versus-second-mode margin of at least 0.04, and a
margin of at least 0.04 over every final association with at least 25 flakes
within four Acus cells. This is still local construction evidence. It is not a
global ownership claim and does not mutate the graph.

The full GTX 1080 run takes 46.9 seconds. Median usable needles increase from
136 in the coarse catalog to 676 after dense extraction. The coarse control
passes 26 of 88 targets and dense Acus passes 35. Twenty-four pass both; two
coarse passes regress under dense extraction. Dense Acus contributes 11 passes
that the control misses, while 44 targets remain below geometry threshold, six
are ownership-ambiguous, and one has an unresolved mode margin.

The 11 apparent recoveries are compared against every stored flake in the exact
target cell with a 6-voxel position, 12-degree normal, and 12-degree fiber gate.
Five match an existing mode owned by another small association. These are useful
evidence for later association recovery but are not missing Acus observations.
Six are new dense modes. Their association scores range from 0.598 to 0.887;
the strongest has 0.553-voxel height, 1.76-degree normal, and 3.09-degree fiber
residuals, with a 0.741 ownership margin. The six occur on six different final
associations, so the result is not a repeated extension of one fragment.

The artifact `fragment-termination-reanalysis-v1.json` contains all skipped
targets, crop bounds and GPU diagnostics, coarse and dense modes, exact residuals,
local competitor scores, stored-cell comparisons, classifications, and the
ranked recovery list. No recovered mode is added to the flake catalog or final
association graph.

This remains a deliberately narrow region-wide construction result, not a sheet
census. Unassociated global branches are outside the carrier-intersection audit,
and no provenance tier is relabeled as independent replication. Output
identities remain sparse exact-coherent surface hypotheses rather than pages or
physical papyrus layers.

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

.venv/bin/python scripts/build-global-boundary-candidates.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/associate-global-branches.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/associate-global-branches.py \
  --root work/cross-scroll-analysis-z512 --local-evidence-only

.venv/bin/python scripts/associate-global-branches.py \
  --root work/cross-scroll-analysis-z512 --accepted-only

.venv/bin/python scripts/associate-global-branches.py \
  --root work/cross-scroll-analysis-z512 --clean-only

.venv/bin/python scripts/census-fragment-terminations.py \
  --root work/cross-scroll-analysis-z512

.venv/bin/python scripts/reanalyze-fragment-terminations.py \
  --root work/cross-scroll-analysis-z512
```

The large NumPy products remain ignored under `work/`. Their summaries contain
content hashes for all generated arrays and input identities. The compact
science baseline is committed under `benchmarks/`.
