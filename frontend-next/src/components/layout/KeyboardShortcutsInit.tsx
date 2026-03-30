"use client";

import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

/**
 * Mounts the global keyboard shortcut handler (Ctrl+1–4 for wizard navigation).
 * Rendered once in the root layout inside the QueryProvider.
 * Renders nothing — purely a side-effect component.
 */
export function KeyboardShortcutsInit() {
  useKeyboardShortcuts();
  return null;
}
