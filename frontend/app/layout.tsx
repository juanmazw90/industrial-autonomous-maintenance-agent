import type { Metadata } from "next";
import { Providers } from "@/components/providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "AMIA — Autonomous Maintenance Intelligence Agent",
  description: "Industrial AI platform for predictive maintenance operations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 h-full antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
