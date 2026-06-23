import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AMIA — Agente de Mantenimiento Industrial",
  description: "Autonomous Maintenance Intelligence Agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body className="bg-gray-950 text-gray-100 h-full">
        <nav className="flex items-center gap-1 border-b border-gray-800 px-4 py-2 text-sm">
          <span className="font-bold text-gray-100 mr-3 tracking-tight">AMIA</span>
          <Link
            href="/"
            className="px-3 py-1 rounded-lg text-gray-400 hover:text-gray-100 hover:bg-gray-800 transition-colors"
          >
            Chat
          </Link>
          <Link
            href="/dashboard"
            className="px-3 py-1 rounded-lg text-gray-400 hover:text-gray-100 hover:bg-gray-800 transition-colors"
          >
            Dashboard
          </Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
