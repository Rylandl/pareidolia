import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rectifier Lab | Local voxel workbench",
  description:
    "Select a voxel and inspect its surrounding scan volume in linked slices and an interactive 3D rendering.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
