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
  FlaskConical,
  DatabaseZap,
  ClipboardCheck,
  ScrollText,
  Wrench,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";

type NavItem =
  | { group: string }
  | { href: string; label: string; icon: React.ElementType };

const NAV: NavItem[] = [
  { group: "Operations" },
  { href: "/dashboard",  label: "Overview",   icon: LayoutDashboard },
  { href: "/assets",     label: "Fleet",      icon: Cpu },
  { href: "/alerts",       label: "Alerts",       icon: BellRing },
  { href: "/work-orders",  label: "Work Orders",  icon: Wrench },
  { group: "AI / ML" },
  { href: "/agents",     label: "AI Agents",  icon: Bot },
  { href: "/models",     label: "ML Models",  icon: FlaskConical },
  { href: "/rag",        label: "RAG",        icon: DatabaseZap },
  { href: "/evaluation", label: "Evaluation", icon: ClipboardCheck },
  { group: "Platform" },
  { href: "/logs",        label: "Logs",       icon: ScrollText },
  { href: "/monitoring",  label: "Services",   icon: Activity },
  { href: "/settings",    label: "Settings",   icon: Settings },
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
      <nav className="flex flex-col gap-0.5 px-3 pt-4 flex-1 overflow-y-auto">
        {NAV.map((item, i) => {
          if ("group" in item) {
            return (
              <p key={i} className="text-[10px] text-gray-600 uppercase tracking-widest px-3 pt-3 pb-1 font-medium">
                {item.group}
              </p>
            );
          }
          const { href, label, icon: Icon } = item;
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
