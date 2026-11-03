/**
 * Results export functionality for Classifast.
 * Supports exporting current classification results to CSV and JSON formats.
 */

export interface ClassificationExportItem {
  code: string;
  name: string;
  score: string;
  scoreRaw?: number;
  level?: string;
  definition?: string;
  url?: string;
}

export interface ClassificationExportPayload {
  query: string;
  classifier: string;
  exportedAt: string;
  results: ClassificationExportItem[];
}

/**
 * Escapes a field for CSV according to RFC 4180.
 * If the field contains commas, double quotes, or newlines,
 * wrap it in double quotes and escape internal double quotes by doubling them.
 */
export function escapeCSVField(value: string | number | undefined | null): string {
  if (value === undefined || value === null) {
    return "";
  }
  const str = String(value);
  if (
    str.includes(",") ||
    str.includes('"') ||
    str.includes("\n") ||
    str.includes("\r")
  ) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Formats a list of classification export items as CSV string with UTF-8 BOM.
 */
export function formatResultsAsCSV(
  items: ClassificationExportItem[],
  _query?: string,
  _classifier?: string,
): string {
  const headers = ["Code", "Name", "Confidence", "Level", "Definition", "URL"];
  const rows = items.map((item) => [
    escapeCSVField(item.code),
    escapeCSVField(item.name),
    escapeCSVField(item.score),
    escapeCSVField(item.level ?? ""),
    escapeCSVField(item.definition ?? ""),
    escapeCSVField(item.url ?? ""),
  ]);

  const csvLines: string[] = [headers.join(",")];
  for (const row of rows) {
    csvLines.push(row.join(","));
  }

  // Prepend UTF-8 BOM so Excel opens UTF-8 characters without garbling
  return "\uFEFF" + csvLines.join("\r\n");
}

/**
 * Formats classification results as a structured JSON string.
 */
export function formatResultsAsJSON(
  items: ClassificationExportItem[],
  query: string = "",
  classifier: string = "",
): string {
  const payload: ClassificationExportPayload = {
    query,
    classifier,
    exportedAt: new Date().toISOString(),
    results: items,
  };
  return JSON.stringify(payload, null, 2);
}

/**
 * Generates an export filename based on classifier, query, and format.
 */
export function generateExportFilename(
  classifier: string,
  query: string,
  extension: "csv" | "json",
): string {
  const safeClassifier = (classifier || "classification")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "");
  const safeQuery = (query || "results")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9_-]/g, "")
    .slice(0, 30);
  const dateStr = new Date().toISOString().slice(0, 10);
  const baseName = safeQuery
    ? `${safeClassifier}-${safeQuery}-${dateStr}`
    : `${safeClassifier}-${dateStr}`;
  return `${baseName}.${extension}`;
}

/**
 * Triggers a file download in the browser.
 */
export function triggerDownload(
  content: string,
  filename: string,
  mimeType: string,
): void {
  const blob = new Blob([content], { type: mimeType });
  const objectUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => {
    window.URL.revokeObjectURL(objectUrl);
  }, 1000);
}

/**
 * Extracts classification results from DOM container (e.g. results section).
 */
export function extractResultsFromDOM(
  container: Document | HTMLElement = document,
): {
  query: string;
  classifier: string;
  items: ClassificationExportItem[];
} {
  const section =
    container.querySelector<HTMLElement>("[data-results-section]") ??
    container.querySelector<HTMLElement>("section[aria-labelledby='results-heading']");

  const query =
    section?.dataset["query"] ||
    container
      .querySelector<HTMLElement>("#results-heading + pre")
      ?.textContent?.replace(/^"|"$/g, "")
      .trim() ||
    "";

  const classifier = section?.dataset["classifier"] || "";

  const itemElements = container.querySelectorAll<HTMLElement>(
    "li[data-classification-result]",
  );

  const items: ClassificationExportItem[] = [];

  itemElements.forEach((el) => {
    const code = el.dataset["code"] || "";
    const name = el.dataset["name"] || "";
    const score = el.dataset["score"] || "";
    const scoreRaw = el.dataset["scoreRaw"]
      ? parseFloat(el.dataset["scoreRaw"])
      : undefined;
    const level = el.dataset["level"] || "";
    const definition = el.dataset["definition"] || "";
    const url = el.dataset["url"] || "";

    if (code) {
      items.push({
        code,
        name,
        score: score ? `${score}%` : "",
        ...(scoreRaw !== undefined ? { scoreRaw } : {}),
        ...(level ? { level } : {}),
        ...(definition ? { definition } : {}),
        ...(url ? { url } : {}),
      });
    }
  });

  return { query, classifier, items };
}

/**
 * ResultExporter handles export button clicks via event delegation.
 */
export class ResultExporter {
  constructor(private readonly signal?: AbortSignal) {
    this.init();
  }

  private init(): void {
    const options: AddEventListenerOptions = {};
    if (this.signal) {
      options.signal = this.signal;
    }

    document.addEventListener(
      "click",
      (event: MouseEvent) => {
        if (!(event.target instanceof Element)) return;
        const button = event.target.closest<HTMLButtonElement>(
          "[data-export-action]",
        );
        if (!button) return;

        const action = button.dataset["exportAction"];
        if (action !== "csv" && action !== "json") return;

        this.handleExport(action);
      },
      options,
    );
  }

  private handleExport(format: "csv" | "json"): void {
    const { query, classifier, items } = extractResultsFromDOM(document);
    if (!items.length) {
      return;
    }

    if (format === "csv") {
      const csvContent = formatResultsAsCSV(items, query, classifier);
      const filename = generateExportFilename(classifier, query, "csv");
      triggerDownload(csvContent, filename, "text/csv;charset=utf-8;");
    } else {
      const jsonContent = formatResultsAsJSON(items, query, classifier);
      const filename = generateExportFilename(classifier, query, "json");
      triggerDownload(jsonContent, filename, "application/json;charset=utf-8;");
    }
  }
}
