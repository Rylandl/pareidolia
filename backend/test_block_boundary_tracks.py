import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.block_sheet_volume import load_block_sheet_payload
from backend.cubical.contracts import sha256_file
from backend.cubical.paired_boundary_surface import (
    PAIRED_BOUNDARY_SURFACE_SCHEMA,
    PAIRED_BOUNDARY_SURFACE_STEM,
)
from backend.cubical.physical_mid_surface import PHYSICAL_MID_SURFACE_STEM


class BlockBoundaryTrackPayloadTests(unittest.TestCase):
    def test_loads_certified_boundary_face_triangles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / f"{PAIRED_BOUNDARY_SURFACE_STEM}.npz"
            np.savez_compressed(
                data_path,
                endpointXYZ=np.asarray(
                    [
                        [12, 23, 34],
                        [14, 23, 34],
                        [12, 25, 34],
                        [14, 25, 34],
                    ],
                    np.float32,
                ),
                endpointCompanion=np.asarray([1, 0, 3, 2], np.int32),
                certifiedFaceTriangleEndpoint=np.asarray(
                    [[0, 1, 2], [1, 3, 2]], np.int32
                ),
                certifiedFaceTriangleComponentId=np.asarray([0, 0], np.int32),
                certifiedFaceTriangleSourceBoundaryTrack=np.asarray(
                    [7, 7], np.int32
                ),
                certifiedFaceBoundaryTriangleIndex=np.asarray([0, 1], np.int32),
                meshTriangleAreaVoxelsSquared=np.asarray([2, 2], np.float32),
                meshTriangleNormalResidualDegrees=np.asarray([3, 5], np.float32),
            )
            manifest_path = root / f"{PAIRED_BOUNDARY_SURFACE_STEM}.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": PAIRED_BOUNDARY_SURFACE_SCHEMA,
                        "version": 1,
                        "state": "complete",
                        "source": {
                            "path": str(root / "volume.npy"),
                            "metadataPath": str(root / "volume.json"),
                            "sourceOriginXYZ": [10, 20, 30],
                            "voxelSizeMicrons": 9.0,
                        },
                        "geometry": {
                            "processingVoxelBounds": {
                                "startXYZ": [0, 0, 0],
                                "stopXYZExclusive": [10, 10, 10],
                            },
                            "ownedWorldBounds": {
                                "startXYZ": [11, 21, 31],
                                "stopXYZExclusive": [19, 29, 39],
                            },
                            "coordinateUnit": "source-voxel",
                        },
                        "counts": {
                            "boundaryEndpointCount": 4,
                            "certifiedBoundaryFaceEndpointCount": 4,
                        },
                        "mesh": {"counts": {"meshEdgeCount": 5}},
                        "data": {
                            "path": data_path.name,
                            "sha256": sha256_file(data_path),
                        },
                    }
                )
            )

            payload = load_block_sheet_payload(root)

            self.assertEqual(
                payload["representation"], "physical-boundary-surface-mesh"
            )
            self.assertEqual(payload["grid"]["originXYZ"], [10.0, 20.0, 30.0])
            self.assertEqual(payload["stats"]["triangleCount"], 2)
            self.assertEqual(payload["stats"]["retainedEdgeCount"], 5)
            self.assertEqual(payload["components"][0]["stableId"], "7")
            self.assertEqual(payload["components"][0]["nodeCount"], 4)
            self.assertEqual(payload["triangles"][1]["vertices"], [1, 3, 2])
            self.assertEqual(payload["vertices"][0], [2.0, 3.0, 4.0])

    def test_loads_boundary_tracks_instead_of_profile_midpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / f"{PHYSICAL_MID_SURFACE_STEM}.npz"
            np.savez_compressed(
                data_path,
                midpointXYZ=np.asarray([[12, 23, 34], [14, 23, 34]], np.float32),
                componentId=np.asarray([0, 0], np.int32),
                nodeKind=np.zeros(2, np.uint8),
                physicalSheetLabel=np.zeros(2, np.int32),
                lowerLocalEvidenceScore=np.asarray([0.8, 0.9], np.float32),
                upperLocalEvidenceScore=np.asarray([0.8, 0.9], np.float32),
                pairCost=np.asarray([0.2, 0.1], np.float32),
                opposingNormalDegrees=np.asarray([2, 3], np.float32),
                macroNormalResidualDegrees=np.asarray([4, 6], np.float32),
                thicknessVoxels=np.asarray([10, 12], np.float32),
                edgeFirstNode=np.asarray([0], np.int32),
                boundaryTrackEndpointXYZ=np.asarray(
                    [[12, 23, 29], [12, 23, 39], [14, 23, 28], [14, 23, 40]],
                    np.float32,
                ),
                boundaryTrackEndpointProfileNode=np.asarray([0, 0, 1, 1], np.int32),
                boundaryTrackComponentId=np.asarray([0, 1, 0, 1], np.int32),
                boundaryTrackLocalSupportDegree=np.asarray([3, 3, 2, 2], np.int32),
                boundaryTrackEdgeFirstEndpoint=np.asarray([0, 1], np.int32),
            )
            manifest_path = root / f"{PHYSICAL_MID_SURFACE_STEM}.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "pareidolia.physical-mid-surface-catalog",
                        "version": 1,
                        "state": "complete",
                        "constructionSchema": "pareidolia.direct-paired-profile-surface",
                        "source": {
                            "path": str(root / "volume.npy"),
                            "metadataPath": str(root / "volume.json"),
                            "voxelSizeMicrons": 9.0,
                        },
                        "geometry": {
                            "ownedWorldBounds": {
                                "startXYZ": [10, 20, 30],
                                "stopXYZExclusive": [20, 30, 50],
                            },
                            "coordinateUnit": "source-voxel",
                        },
                        "counts": {"selectedCandidateCount": 2},
                        "selection": {"spatialKeyCount": 2},
                        "boundaryTracks": {
                            "counts": {"componentCount": 2},
                            "collisionGuard": {
                                "columnConflictRejectedEdgeCount": 7
                            },
                        },
                        "data": {
                            "path": data_path.name,
                            "sha256": sha256_file(data_path),
                        },
                    }
                )
            )

            payload = load_block_sheet_payload(root)

            self.assertEqual(payload["representation"], "physical-boundary-track-graph")
            self.assertEqual(payload["stats"]["nodeCount"], 4)
            self.assertEqual(payload["stats"]["componentCount"], 2)
            self.assertEqual(payload["stats"]["retainedEdgeCount"], 2)
            self.assertEqual(payload["stats"]["columnConflictRejectedEdgeCount"], 7)
            self.assertEqual(payload["components"][0]["nodeCount"], 2)
            self.assertEqual(payload["interfaceNodes"][0], [2.0, 3.0, -1.0, 1, 0])


if __name__ == "__main__":
    unittest.main()
