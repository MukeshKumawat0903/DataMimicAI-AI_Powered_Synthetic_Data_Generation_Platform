import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { ColumnInfo, ColumnType, DatasetResponse, AppError } from "@/lib/api/types";

// Backend response shape from GET /eda/get-data/{file_id}
interface BackendDatasetResponse {
  data: Record<string, unknown>[];
  columns: string[];
  shape: [number, number];
  dtypes: Record<string, string>;
}

function toColumnType(dtype: string, uniqueCount: number, rowCount: number): ColumnType {
  if (/int|float|uint/.test(dtype)) return "numeric";
  if (dtype.includes("datetime")) return "datetime";
  if (dtype === "bool" || dtype === "category") return "categorical";
  if (dtype === "object") {
    // Mostly-unique string columns (names, IDs, notes) behave like free text
    return rowCount > 0 && uniqueCount / rowCount > 0.5 ? "text" : "categorical";
  }
  return "unknown";
}

function mapResponse(fileId: string, res: BackendDatasetResponse): DatasetResponse {
  const [numRows, numColumns] = res.shape;
  const rows = res.data;

  const columns: ColumnInfo[] = res.columns.map((name) => {
    const values = rows.map((r) => r[name]);
    const missing = values.filter((v) => v === null || v === undefined).length;
    const uniqueCount = new Set(values.filter((v) => v !== null && v !== undefined)).size;
    return {
      name,
      type: toColumnType(res.dtypes[name] ?? "", uniqueCount, rows.length),
      missing_pct: rows.length ? (missing / rows.length) * 100 : 0,
      unique_count: uniqueCount,
    };
  });

  return {
    file_id: fileId,
    columns,
    num_rows: numRows,
    num_columns: numColumns,
    sample_data: rows,
  };
}

export function useDataset(fileId: string | null) {
  return useQuery<DatasetResponse, AppError>({
    queryKey: ["dataset", fileId],
    queryFn: async () => {
      const res = await apiClient.get<BackendDatasetResponse>(
        `/eda/get-data/${fileId}`
      );
      return mapResponse(fileId as string, res.data);
    },
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 1,
  });
}
