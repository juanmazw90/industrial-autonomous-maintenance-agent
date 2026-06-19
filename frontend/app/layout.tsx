import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AMIA — Agente de Mantenimiento Industrial",
  description: "Autonomous Maintenance Intelligence Agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body className="bg-gray-950 text-gray-100 h-full">{children}</body>
    </html>
  );
}
