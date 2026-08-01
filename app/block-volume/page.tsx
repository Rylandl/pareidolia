import type { Metadata } from "next";
import { BlockVolumeExplorer } from "../BlockVolumeExplorer";

export const metadata: Metadata = {
  title: "Current surface components | Rectifier Lab",
  description: "Inspect the current cubical surface components and experimental completions inside their aligned CT block.",
};

export default function BlockVolumePage() {
  return <BlockVolumeExplorer />;
}
