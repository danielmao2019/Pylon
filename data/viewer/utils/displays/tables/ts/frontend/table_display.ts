import type { LeafVNode } from "web/reconcile/reconcile";
import type { TableDisplayResponse } from "./types/display_response";

export function renderTableDisplay({
  displayResponse,
}: {
  displayResponse: TableDisplayResponse;
}): LeafVNode {
  return {
    kind: "leaf",
    key: displayResponse.url ?? `table:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      if (displayResponse.url === null) {
        const placeholder = document.createElement("div");
        placeholder.className = "placeholder-surface";
        placeholder.textContent = "Placeholder for a benchmark result that is not materialized yet.";
        return placeholder;
      }

      const tableWrap = document.createElement("div");
      tableWrap.className = "table-wrap";
      tableWrap.textContent = "Loading table";
      void loadTableDisplay({
        tableWrap,
        displayResponse,
      });
      return tableWrap;
    },
  };
}

async function loadTableDisplay({
  tableWrap,
  displayResponse,
}: {
  tableWrap: HTMLDivElement;
  displayResponse: TableDisplayResponse;
}): Promise<void> {
  if (displayResponse.url === null) {
    throw new Error("table display response url is null");
  }
  const response = await fetch(displayResponse.url);
  if (!response.ok) {
    tableWrap.textContent = `Unable to load table: HTTP ${response.status}`;
    return;
  }
  const text = await response.text();
  const rows = readRowsFromArtifact({
    text,
    url: displayResponse.url,
  });
  tableWrap.replaceChildren(renderRows({ rows }));
}

function renderRows({ rows }: { rows: Record<string, string>[] }): HTMLElement {
  const table = document.createElement("table");
  const discoveredColumns = Array.from(
    new Set(rows.flatMap((row) => Object.keys(row))),
  );
  const preferredColumns = ["Question", "GT Answer", "Pred Answer", "Judgement"];
  const columns = [
    ...preferredColumns.filter((column) => discoveredColumns.includes(column)),
    ...discoveredColumns.filter((column) => !preferredColumns.includes(column)),
  ];

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const column of columns) {
    const cell = document.createElement("th");
    cell.textContent = column;
    cell.dataset.column = column;
    headerRow.append(cell);
  }
  thead.append(headerRow);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tableRow = document.createElement("tr");
    for (const column of columns) {
      const cell = document.createElement("td");
      cell.textContent = row[column] ?? "";
      cell.dataset.column = column;
      if (column === "Judgement") {
        cell.className = `judgement-cell judgement-cell--${(row[column] ?? "").toLowerCase()}`;
      }
      tableRow.append(cell);
    }
    tbody.append(tableRow);
  }

  table.append(thead, tbody);
  return table;
}

function readRowsFromArtifact({
  text,
  url,
}: {
  text: string;
  url: string;
}): Record<string, string>[] {
  if (url.includes(".jsonl")) {
    return text
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
      .slice(0, 12)
      .map((line) => normalizeRow(JSON.parse(line)));
  }
  const parsed: unknown = JSON.parse(text);
  if (Array.isArray(parsed)) {
    return parsed.slice(0, 12).map(normalizeRow);
  }
  if (isRecord(parsed)) {
    return [normalizeRow(parsed)];
  }
  return [];
}

function normalizeRow(value: unknown): Record<string, string> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, String(item)]),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  return true;
}
