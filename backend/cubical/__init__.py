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
from .contracts import (
    ExtractionTileSpec,
    RawAcusSettings,
    ReconstructionWindow,
    ShardSpec,
    VolumeSource,
    VoxelBounds,
    extraction_tiles_for_shard,
    plan_extraction_tiles,
    plan_shards,
)
from .evidence import CellEvidenceTable
from .gaps import GapCensus, GapTraceRecord, analyze_component_gaps
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
from .pipeline import run_raw_acus_pipeline
from .raw_acus import AcusCalibration, NeedleTable
from .selection import (
    ConfigurationSelection,
    configuration_options,
    optimize_configurations,
)
from .stratigraphy import ConfigurationTable, LayerModeTable
from .topology import GridEdge, GridFace, GridSpec
from .synthetic import SyntheticScene, SyntheticStackSettings, generate_synthetic_stack
from .tables import PatchTable, read_patch_shard, write_patch_shard

__all__ = [
    "ClippedPatch",
    "DegeneratePlaneIntersection",
    "DeferredJoin",
    "EdgeCrossing",
    "EndpointAgreement",
    "ExtractionTileSpec",
    "FaceTrace",
    "GridEdge",
    "GridFace",
    "GridSpec",
    "GapCensus",
    "GapTraceRecord",
    "BlockBounds",
    "AcusCalibration",
    "AcusAdapterSettings",
    "AcusWindowScene",
    "PlaneEstimate",
    "RawAcusSettings",
    "ReconstructionWindow",
    "ShardSpec",
    "VolumeSource",
    "VoxelBounds",
    "NeedleTable",
    "CellEvidenceTable",
    "ConfigurationTable",
    "LayerModeTable",
    "ConfigurationSelection",
    "configuration_options",
    "FaceAlignment",
    "TraceMatch",
    "TraceMatchSettings",
    "SurfaceBlock",
    "SyntheticScene",
    "SyntheticStackSettings",
    "PatchTable",
    "align_face_patches",
    "analyze_component_gaps",
    "assemble_surface_block",
    "assemble_surface_hierarchy",
    "axial_angle_radians",
    "clip_plane_to_cell",
    "generate_synthetic_stack",
    "match_face_traces",
    "load_acus_flake_window",
    "merge_surface_blocks",
    "optimize_configurations",
    "extraction_tiles_for_shard",
    "plan_extraction_tiles",
    "plan_shards",
    "read_patch_shard",
    "write_patch_shard",
    "run_raw_acus_pipeline",
]
