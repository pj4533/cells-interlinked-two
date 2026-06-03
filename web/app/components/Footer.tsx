"use client";

// Footer nav. Polls autoresearch status: while the hunt is running it owns M,
// so the M-driving pages (interrogate / chat / trip / autorun / pairs /
// baseline) are locked out and greyed; read-only pages stay live, and the
// autoresearch link lights up.

import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchAutoresearchState } from "@/lib/autoresearch";
import { fetchDmtState } from "@/lib/autoresearch-dmt";

// Pages that drive M — disabled while EITHER autoresearch holds it. The two AR
// monitor pages stay navigable (they're read-only viewers); the backend enforces
// that only one AR can run at a time, and its start button errors otherwise.
const M_LOCKED = new Set(["/", "/chat", "/trip", "/autorun", "/pairs", "/baseline"]);

const LINKS: { href: string; label: string; cls: string }[] = [
  { href: "/", label: "cells interlinked", cls: "hover:text-amber" },
  { href: "/chat", label: "chat", cls: "hover:text-cyan" },
  { href: "/trip", label: "trip", cls: "hover:text-cyan" },
  { href: "/autoresearch", label: "off-manifold AR", cls: "hover:text-cyan" },
  { href: "/autoresearch-dmt", label: "DMT AR", cls: "hover:text-cyan" },
  { href: "/archive", label: "archive", cls: "hover:text-amber" },
  { href: "/pairs", label: "pairs", cls: "hover:text-amber" },
  { href: "/autorun", label: "autorun", cls: "hover:text-cyan" },
  { href: "/journal", label: "journal", cls: "hover:text-cyan" },
  { href: "/baseline", label: "baseline", cls: "hover:text-amber-dim" },
  { href: "/fine-print", label: "read the fine print", cls: "hover:text-amber" },
];

export default function Footer() {
  const [offRun, setOffRun] = useState(false);
  const [dmtRun, setDmtRun] = useState(false);
  const anyRun = offRun || dmtRun;

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const [off, dmt] = await Promise.all([fetchAutoresearchState(), fetchDmtState()]);
      if (!alive) return;
      setOffRun(!!off?.running);
      setDmtRun(!!dmt?.running);
    };
    poll();
    const t = setInterval(poll, 4000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <footer className="border-t border-rule bg-bg-soft py-3 px-6 text-xs text-text-dim flex justify-between items-center gap-3 flex-wrap">
      <div className="space-x-6">
        {LINKS.map((l) => {
          const locked = anyRun && M_LOCKED.has(l.href);
          if (locked) {
            return (
              <span key={l.href} className="opacity-40 cursor-not-allowed" title="locked — autoresearch is running">
                {l.label}
              </span>
            );
          }
          const active = (offRun && l.href === "/autoresearch") || (dmtRun && l.href === "/autoresearch-dmt");
          return (
            <Link
              key={l.href}
              href={l.href}
              className={`${l.cls} ${active ? "text-cyan cyan-glow" : ""}`}
            >
              {l.label}
            </Link>
          );
        })}
      </div>
      <div className="text-right opacity-60 italic">
        {anyRun
          ? <span className="text-cyan not-italic font-display tracking-widest text-[10px]">◉ {dmtRun ? "DMT AR" : "OFF-MANIFOLD AR"} RUNNING — other instruments locked</span>
          : "more human than human is our motto. © Tyrell Corporation 2019"}
      </div>
    </footer>
  );
}
