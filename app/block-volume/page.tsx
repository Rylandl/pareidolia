import type { Metadata } from "next";
import { BlockVolumeExplorer } from "../BlockVolumeExplorer";

export const metadata: Metadata = {
  title: "Physical boundary surfaces | Rectifier Lab",
  description: "Inspect collision-safe signed papyrus boundary meshes inside their aligned CT block.",
};

export default function BlockVolumePage() {
  return <BlockVolumeExplorer />;
}
