"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Cpu,
  Bot,
  BellRing,
  Activity,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard", label: "Operations", icon: LayoutDashboard },
  { href: "/assets",    label: "Fleet",      icon: Cpu },
  { href: "/agents",    label: "AI Agents",  icon: Bot },
  { href: "/alerts",    label: "Alerts",     icon: BellRing },
  { href: "/monitoring",label: "Services",   icon: Activity },
];

function Sidebar() {
  const path = usePathname();
  return (
    <aside className="flex flex-col w-56 shrink-0 border-r border-gray-800 bg-gray-950 h-screen sticky top-0">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-5 py-5 border-b border-gray-800">
        <div className="w-6 h-6 rounded bg-indigo-600 flex items-center justify-center">
          <span className="text-white text-xs font-bold">A</span>
        </div>
        <span className="font-semibold text-gray-100 tracking-tight text-sm">AMIA Platform</span>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 px-3 pt-4 flex-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = path === href || path.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors group",
                active
                  ? "bg-indigo-950/60 text-indigo-300 border border-indigo-800/50"
                  : "text-gray-500 hover:text-gray-200 hover:bg-gray-800/60"
              )}
            >
              <Icon size={15} className={active ? "text-indigo-400" : "text-gray-600 group-hover:text-gray-400"} />
              {label}
              {active && <ChevronRight size={12} className="ml-auto text-indigo-500" />}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-gray-800">
        <Link href="/" className="text-xs text-gray-600 hover:text-gray-400 transition-colors">
          ← Back to Chat
        </Link>
      </div>
    </aside>
  );
}

export default function PlatformLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-gray-950">
        {children}
      </main>
    </div>
  );
}
