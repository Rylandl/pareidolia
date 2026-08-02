import type { Metadata } from "next";
import { BlockVolumeExplorer } from "../BlockVolumeExplorer";

export const metadata: Metadata = {
  title: "Physical boundary tracks | Rectifier Lab",
  description: "Inspect collision-safe physical papyrus boundary tracks inside their aligned CT block.",
};

export default function BlockVolumePage() {
  return <BlockVolumeExplorer />;
}
