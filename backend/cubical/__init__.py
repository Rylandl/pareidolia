"""Dataset-independent cubical surface reconstruction primitives."""

from .acus_adapter import (
    AcusAdapterSettings,
    AcusWindowScene,
    load_acus_flake_window,
)
from .block import (
    BlockBounds,
    DeferredJoin,
    SurfaceBlock,
    assemble_surface_hierarchy,
    assemble_surface_block,
    merge_surface_blocks,
)
from .geometry import (
    ClippedPatch,
    DegeneratePlaneIntersection,
    EdgeCrossing,
    FaceTrace,
    PlaneEstimate,
    axial_angle_radians,
    clip_plane_to_cell,
)
from .matching import (
    EndpointAgreement,
    FaceAlignment,
    TraceMatch,
    TraceMatchSettings,
    align_face_patches,
    match_face_traces,
)
from .topology import GridEdge, GridFace, GridSpec
from .synthetic import SyntheticScene, SyntheticStackSettings, generate_synthetic_stack
from .tables import PatchTable, read_patch_shard, write_patch_shard

__all__ = [
    "ClippedPatch",
    "DegeneratePlaneIntersection",
    "DeferredJoin",
    "EdgeCrossing",
    "EndpointAgreement",
    "FaceTrace",
    "GridEdge",
    "GridFace",
    "GridSpec",
    "BlockBounds",
    "AcusAdapterSettings",
    "AcusWindowScene",
    "PlaneEstimate",
    "FaceAlignment",
    "TraceMatch",
    "TraceMatchSettings",
    "SurfaceBlock",
    "SyntheticScene",
    "SyntheticStackSettings",
    "PatchTable",
    "align_face_patches",
    "assemble_surface_block",
    "assemble_surface_hierarchy",
    "axial_angle_radians",
    "clip_plane_to_cell",
    "generate_synthetic_stack",
    "match_face_traces",
    "load_acus_flake_window",
    "merge_surface_blocks",
    "read_patch_shard",
    "write_patch_shard",
]
