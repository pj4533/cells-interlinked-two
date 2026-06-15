"use client";

// Footer nav. Polls DMT autoresearch status: while the hunt runs it owns M, so
// the M-driving pages (chat / trip) are locked out and greyed; the DMT AR
// monitor (a read-only viewer) stays live and its link lights up.

import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchDmtState } from "@/lib/autoresearch-dmt";

// Pages that drive M — disabled while the DMT hunt holds it. The DMT AR monitor
// stays navigable (read-only viewer); its start button errors if compute is busy.
const M_LOCKED = new Set(["/", "/chat", "/trip"]);

const LINKS: { href: string; label: string; cls: string }[] = [
  { href: "/", label: "cells interlinked", cls: "hover:text-amber" },
  { href: "/chat", label: "chat", cls: "hover:text-cyan" },
  { href: "/trip", label: "trip", cls: "hover:text-cyan" },
  { href: "/autoresearch-dmt", label: "DMT AR", cls: "hover:text-cyan" },
  { href: "/archive", label: "archive", cls: "hover:text-amber" },
  { href: "/journal", label: "journal", cls: "hover:text-cyan" },
];

export default function Footer() {
  const [dmtRun, setDmtRun] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const dmt = await fetchDmtState();
      if (!alive) return;
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
          const locked = dmtRun && M_LOCKED.has(l.href);
          if (locked) {
            return (
              <span key={l.href} className="opacity-40 cursor-not-allowed" title="locked — DMT autoresearch is running">
                {l.label}
              </span>
            );
          }
          const active = dmtRun && l.href === "/autoresearch-dmt";
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
        {dmtRun
          ? <span className="text-cyan not-italic font-display tracking-widest text-[10px]">◉ DMT AR RUNNING — chat &amp; trip locked</span>
          : "more human than human is our motto. © Tyrell Corporation 2019"}
      </div>
    </footer>
  );
}
