import type { Metadata } from "next";
import { CrossScrollExplorer } from "../CrossScrollExplorer";

export const metadata: Metadata = {
  title: "Cross-scroll field | Rectifier Lab",
  description: "Explore full-resolution axial slices of the Acus normal and page-tangent field.",
};

export default function CrossScrollPage() {
  return <CrossScrollExplorer />;
}
