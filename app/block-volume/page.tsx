import type { Metadata } from "next";
import { BlockVolumeExplorer } from "../BlockVolumeExplorer";

export const metadata: Metadata = {
  title: "Solved block volume | Rectifier Lab",
  description: "Inspect the source CT volume and every retained Acus sheet patch in one orbitable block.",
};

export default function BlockVolumePage() {
  return <BlockVolumeExplorer />;
}
