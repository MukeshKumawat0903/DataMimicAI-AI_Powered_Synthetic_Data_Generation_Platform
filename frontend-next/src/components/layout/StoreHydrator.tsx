"use client";

import { useEffect } from "react";
import { useWorkspaceStore } from "@/lib/store/workspace";

/** Load the saved session after the first render, so server and client HTML match */
export function StoreHydrator() {
  useEffect(() => {
    useWorkspaceStore.persist.rehydrate();
  }, []);
  return null;
}
