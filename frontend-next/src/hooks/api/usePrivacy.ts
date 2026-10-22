import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { PrivacyResponse } from "@/lib/api/types";

// Backend response shape from POST /eda/pii-scan-fast and /eda/pii-scan-deep
interface BackendPrivacyResponse {
  scan_type?: "fast" | "deep";
  results?: {
    detections?: Array<{
      column: string;
      pii_types?: string[];
      entity_type?: string;
      confidence?: number;
      samples?: string[];
    }>;
  };
}

function mapResponse(res: BackendPrivacyResponse, mode: "fast" | "deep"): PrivacyResponse {
  return {
    scan_mode: res.scan_type ?? mode,
    pii_results: (res.results?.detections ?? []).map((d) => ({
      column: d.column,
      pii_type: d.pii_types?.join(", ") || d.entity_type || "Unknown",
      confidence: d.confidence ?? 0,
      sample_masked: d.samples?.[0] ?? "—",
    })),
    // PII scans don't compute k-anonymity (that's /eda/compute-k-anonymity)
    k_anonymity: undefined,
  };
}

export function usePrivacy(fileId: string | null, mode: "fast" | "deep" = "fast") {
  return useQuery<PrivacyResponse>({
    queryKey: ["privacy", fileId, mode],
    queryFn: () => {
      if (mode === "deep") {
        return apiClient
          .post<BackendPrivacyResponse>(`/eda/pii-scan-deep`, null, {
            params: { file_id: fileId },
          })
          .then((r) => mapResponse(r.data, mode));
      }
      return apiClient
        .post<BackendPrivacyResponse>(`/eda/pii-scan-fast/${fileId}`, null)
        .then((r) => mapResponse(r.data, mode));
    },
    enabled: !!fileId,
    staleTime: 10 * 60 * 1000,
  });
}
