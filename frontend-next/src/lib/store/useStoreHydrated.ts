"use client";

import { useEffect, useState } from "react";
import { useWorkspaceStore } from "./workspace";

/** True once the saved session (sessionStorage) has been loaded into the store */
export function useStoreHydrated() {
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => {
    const unsub = useWorkspaceStore.persist.onFinishHydration(() => setHydrated(true));
    setHydrated(useWorkspaceStore.persist.hasHydrated());
    return unsub;
  }, []);
  return hydrated;
}
