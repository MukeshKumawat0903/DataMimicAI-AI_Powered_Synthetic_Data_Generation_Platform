"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

const ROUTES = ["/upload", "/explore", "/generate", "/validate"] as const;

/**
 * Registers Ctrl+1–4 keyboard shortcuts for wizard step navigation.
 * Call once at the app root (e.g. inside a layout client component).
 */
export function useKeyboardShortcuts() {
  const router = useRouter();

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Ignore when user is typing in an input, textarea, or contenteditable
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement).isContentEditable) {
        return;
      }

      if (e.ctrlKey && !e.shiftKey && !e.altKey && !e.metaKey) {
        const idx = Number(e.key) - 1;
        if (idx >= 0 && idx < ROUTES.length) {
          e.preventDefault();
          router.push(ROUTES[idx]);
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [router]);
}
