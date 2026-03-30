"use client";

import { useState, useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type VisibilityState,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChevronDown, ChevronUp, ChevronsUpDown, Columns3 } from "lucide-react";
import type { ColumnInfo, ColumnType } from "@/lib/api/types";
import { cn } from "@/lib/utils";

// ─── Imports for DropdownMenu (shadcn uses radix under the hood) ─────────────
// We need to install it if not already present. The shadcn dropdown-menu
// component is separate from select; add it inline here as a lightweight
// version using the card-based approach.
// NOTE: DropdownMenu is from radix via shadcn — add the component if missing.

// ─── Column type badge colours ────────────────────────────────────────────────

const TYPE_VARIANT: Record<ColumnType, string> = {
  numeric: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  categorical:
    "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300",
  datetime:
    "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300",
  text: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  unknown: "bg-zinc-100 text-zinc-500",
};

// ─── Sort icon helper ─────────────────────────────────────────────────────────

function SortIcon({ sorted }: { sorted: false | "asc" | "desc" }) {
  if (sorted === "asc") return <ChevronUp className="ml-1 h-3.5 w-3.5 inline" />;
  if (sorted === "desc") return <ChevronDown className="ml-1 h-3.5 w-3.5 inline" />;
  return (
    <ChevronsUpDown className="ml-1 h-3.5 w-3.5 inline text-muted-foreground/50" />
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

interface DataTableProps {
  data: Record<string, unknown>[];
  columnInfo?: ColumnInfo[];
  /** Default page size */
  pageSize?: number;
}

export function DataTable({
  data,
  columnInfo = [],
  pageSize = 20,
}: DataTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});

  // P2.1: memoize derived structures to avoid re-computing on every render
  const typeMap = useMemo(
    () =>
      Object.fromEntries(columnInfo.map((c) => [c.name, c.type])) as Record<
        string,
        ColumnType
      >,
    [columnInfo]
  );

  // Derive columns dynamically from first data row
  const colKeys = useMemo(
    () => (data.length > 0 ? Object.keys(data[0]) : []),
    [data]
  );

  const columns: ColumnDef<Record<string, unknown>>[] = useMemo(
    () =>
      colKeys.map((key) => ({
        id: key,
        accessorKey: key,
        header: ({ column }) => (
          <button
            className="flex items-center text-left font-medium text-xs uppercase tracking-wide"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
            // P3.4: descriptive aria-label for sort state
            aria-label={`Sort by ${key}${
              column.getIsSorted() === "asc"
                ? ", currently ascending"
                : column.getIsSorted() === "desc"
                ? ", currently descending"
                : ", not sorted"
            }`}
          >
            <span>{key}</span>
            <SortIcon sorted={column.getIsSorted()} />
            {typeMap[key] && (
              <span
                className={cn(
                  "ml-2 rounded px-1 py-0.5 text-[10px] font-semibold",
                  TYPE_VARIANT[typeMap[key]] ?? TYPE_VARIANT.unknown
                )}
              >
                {typeMap[key]}
              </span>
            )}
          </button>
        ),
        cell: ({ getValue }) => {
          const v = getValue();
          return (
            <span className="text-sm text-foreground/80 truncate max-w-[180px] block">
              {v === null || v === undefined ? (
                <span className="text-muted-foreground italic text-xs">null</span>
              ) : (
                String(v)
              )}
            </span>
          );
        },
      })),
    [colKeys, typeMap]
  );

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter, columnVisibility },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    initialState: { pagination: { pageSize } },
  });

  if (data.length === 0) {
    return (
      <div className="rounded-lg border p-10 text-center text-sm text-muted-foreground">
        No data available.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <Input
          placeholder="Search all columns…"
          value={globalFilter}
          onChange={(e) => setGlobalFilter(e.target.value)}
          className="max-w-xs h-8 text-sm"
        />

        {/* Column visibility toggle */}
        <DropdownMenuColumnToggle table={table} />
      </div>

      {/* Table */}
      <div className="rounded-md border overflow-x-auto">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((hg) => (
              <TableRow key={hg.id} className="bg-muted/50">
                {hg.headers.map((header) => (
                  <TableHead key={header.id} className="whitespace-nowrap">
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.length === 0 ? (
              // P2.1 + filter UX: show helpful message when filter returns no rows
              <TableRow>
                <TableCell
                  colSpan={colKeys.length}
                  className="py-10 text-center text-sm text-muted-foreground"
                >
                  No rows match your filter.
                </TableCell>
              </TableRow>
            ) : (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="hover:bg-muted/30">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className="py-1.5">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          Showing{" "}
          {table.getState().pagination.pageIndex *
            table.getState().pagination.pageSize +
            1}
          –
          {Math.min(
            (table.getState().pagination.pageIndex + 1) *
              table.getState().pagination.pageSize,
            table.getFilteredRowModel().rows.length
          )}{" "}
          of {table.getFilteredRowModel().rows.length} rows
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            ← Prev
          </Button>
          <span className="px-2 text-xs">
            Page {table.getState().pagination.pageIndex + 1} /{" "}
            {table.getPageCount()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next →
          </Button>
        </div>
      </div>
    </div>
  );
}

// ─── Column visibility dropdown ───────────────────────────────────────────────

// Inline lightweight dropdown to avoid needing to install the full
// shadcn dropdown-menu component just for this feature.
function DropdownMenuColumnToggle({
  table,
}: {
  table: ReturnType<typeof useReactTable<Record<string, unknown>>>;
}) {
  const [open, setOpen] = useState(false);
  const hideable = table
    .getAllColumns()
    .filter((c) => c.getCanHide());

  return (
    <div className="relative">
      <Button
        variant="outline"
        size="sm"
        className="h-8 gap-1 text-xs"
        onClick={() => setOpen((o) => !o)}
      >
        <Columns3 className="h-3.5 w-3.5" />
        Columns
      </Button>
      {open && (
        <div className="absolute right-0 z-50 mt-1 w-44 rounded-md border bg-popover p-1 shadow-md">
          <p className="px-2 py-1 text-xs font-semibold text-muted-foreground">
            Toggle columns
          </p>
          <div className="max-h-60 overflow-y-auto">
            {hideable.map((column) => (
              <label
                key={column.id}
                className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-accent"
              >
                <input
                  type="checkbox"
                  checked={column.getIsVisible()}
                  onChange={() => column.toggleVisibility()}
                  className="h-3.5 w-3.5 accent-primary"
                />
                <span className="truncate">{column.id}</span>
              </label>
            ))}
          </div>
          <div className="border-t mt-1 pt-1 px-2">
            <button
              className="text-xs text-muted-foreground hover:text-foreground w-full text-left py-1"
              onClick={() => setOpen(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
