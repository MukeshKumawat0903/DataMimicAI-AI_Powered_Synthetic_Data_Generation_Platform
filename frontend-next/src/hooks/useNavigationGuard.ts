"use client";

import { useEffect } from "react";

/**
 * Blocks the browser's beforeunload prompt to warn users before
 * navigating away while a critical async operation is in progress.
 *
 * @param isActive  When true, the guard is active (e.g. generate.isPending).
 */
export function useNavigationGuard(isActive: boolean) {
  useEffect(() => {
    if (!isActive) return;

    function handler(e: BeforeUnloadEvent) {
      // Modern browsers show a generic message regardless of the string set
      e.preventDefault();
    }

    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isActive]);
}
