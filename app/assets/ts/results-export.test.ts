import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  escapeCSVField,
  formatResultsAsCSV,
  formatResultsAsJSON,
  generateExportFilename,
  extractResultsFromDOM,
  ResultExporter,
  triggerDownload,
  type ClassificationExportItem,
} from "./results-export";

describe("results-export", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  describe("escapeCSVField", () => {
    it("returns empty string for null and undefined", () => {
      expect(escapeCSVField(null)).toBe("");
      expect(escapeCSVField(undefined)).toBe("");
    });

    it("returns plain string when no special characters", () => {
      expect(escapeCSVField("Laptop computers")).toBe("Laptop computers");
      expect(escapeCSVField(43211507)).toBe("43211507");
    });

    it("wraps and escapes values containing commas", () => {
      expect(escapeCSVField("Apple, Inc.")).toBe('"Apple, Inc."');
    });

    it("escapes quotes by doubling them", () => {
      expect(escapeCSVField('Display 15"')).toBe('"Display 15"""');
    });

    it("escapes newlines", () => {
      expect(escapeCSVField("Line 1\nLine 2")).toBe('"Line 1\nLine 2"');
      expect(escapeCSVField("Line 1\r\nLine 2")).toBe('"Line 1\r\nLine 2"');
    });

    it("neutralizes formula execution prefixes (=, +, -, @, tab, cr)", () => {
      expect(escapeCSVField("=cmd|' /C calc'!A0")).toBe("'=cmd|' /C calc'!A0");
      expect(escapeCSVField("+SUM(1+1)")).toBe("'+SUM(1+1)");
      expect(escapeCSVField("-10% discount")).toBe("'-10% discount");
      expect(escapeCSVField("@SUM(A1:A5)")).toBe("'@SUM(A1:A5)");
      expect(escapeCSVField("\tformula")).toBe("'\tformula");
      expect(escapeCSVField("=SUM(A1, B1)")).toBe('"\'=SUM(A1, B1)"');
    });
  });

  describe("formatResultsAsCSV", () => {
    it("formats empty items with header and BOM", () => {
      const csv = formatResultsAsCSV([]);
      expect(csv.startsWith("\uFEFF")).toBe(true);
      expect(csv).toContain("Code,Name,Confidence,Level,Definition,URL");
    });

    it("formats multiple items correctly", () => {
      const items: ClassificationExportItem[] = [
        {
          code: "43211507",
          name: "Laptop computers",
          score: "94.20%",
          level: "Commodity",
          definition: "Portable battery-powered computers",
          url: "https://example.com/43211507",
        },
        {
          code: "43211503",
          name: "Notebook computers, handheld",
          score: "85.10%",
          definition: "Handheld devices",
        },
      ];

      const csv = formatResultsAsCSV(items);
      expect(csv).toContain(
        "43211507,Laptop computers,94.20%,Commodity,Portable battery-powered computers,https://example.com/43211507",
      );
      expect(csv).toContain(
        '43211503,"Notebook computers, handheld",85.10%,,Handheld devices,',
      );
    });
  });

  describe("formatResultsAsJSON", () => {
    it("serializes payload with metadata and results", () => {
      const items: ClassificationExportItem[] = [
        {
          code: "43211507",
          name: "Laptop computers",
          score: "94.20%",
          scoreRaw: 0.942,
        },
      ];

      const jsonStr = formatResultsAsJSON(items, "MacBook", "UNSPSC");
      const parsed = JSON.parse(jsonStr);

      expect(parsed.query).toBe("MacBook");
      expect(parsed.classifier).toBe("UNSPSC");
      expect(parsed.results).toHaveLength(1);
      expect(parsed.results[0].code).toBe("43211507");
      expect(parsed.results[0].score).toBe("94.20%");
      expect(parsed.results[0].scoreRaw).toBe(0.942);
      expect(parsed.exportedAt).toBeDefined();
    });
  });

  describe("generateExportFilename", () => {
    it("generates filename with classifier and sanitized query", () => {
      const filename = generateExportFilename("UNSPSC", "Apple MacBook Air!", "csv");
      expect(filename.startsWith("unspsc-apple-macbook-air-")).toBe(true);
      expect(filename.endsWith(".csv")).toBe(true);
    });

    it("falls back to defaults when classifier or query is empty", () => {
      const filename = generateExportFilename("", "", "json");
      expect(filename.startsWith("classification-")).toBe(true);
      expect(filename.endsWith(".json")).toBe(true);
    });
  });

  describe("extractResultsFromDOM", () => {
    it("extracts query, classifier and items from DOM", () => {
      document.body.innerHTML = `
        <section aria-labelledby="results-heading" data-results-section data-query="iPhone 15" data-classifier="UNSPSC">
          <figure>
            <figcaption id="results-heading">Results for:</figcaption>
            <pre>"iPhone 15"</pre>
          </figure>
          <ul>
            <li role="listitem"
                data-classification-result
                data-code="43191501"
                data-name="Mobile phones"
                data-score="98.50"
                data-score-raw="0.985"
                data-level="Commodity"
                data-definition="Cellular phones"
                data-url="https://example.com/43191501">
            </li>
          </ul>
        </section>
      `;

      const result = extractResultsFromDOM(document);
      expect(result.query).toBe("iPhone 15");
      expect(result.classifier).toBe("UNSPSC");
      expect(result.items).toHaveLength(1);
      expect(result.items[0]).toEqual({
        code: "43191501",
        name: "Mobile phones",
        score: "98.50%",
        scoreRaw: 0.985,
        level: "Commodity",
        definition: "Cellular phones",
        url: "https://example.com/43191501",
      });
    });

    it("handles DOM with no results", () => {
      document.body.innerHTML = "<div>No results</div>";
      const result = extractResultsFromDOM(document);
      expect(result.items).toHaveLength(0);
      expect(result.query).toBe("");
    });
  });

  describe("triggerDownload", () => {
    it("creates blob and clicks anchor element", () => {
      const createObjectURLMock = vi.fn().mockReturnValue("blob:http://localhost/test");
      const revokeObjectURLMock = vi.fn();
      window.URL.createObjectURL = createObjectURLMock;
      window.URL.revokeObjectURL = revokeObjectURLMock;

      const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

      triggerDownload("test-content", "test.csv", "text/csv");

      expect(createObjectURLMock).toHaveBeenCalled();
      expect(clickSpy).toHaveBeenCalled();

      clickSpy.mockRestore();
    });
  });

  describe("ResultExporter", () => {
    it("triggers CSV export when clicking [data-export-action='csv']", () => {
      document.body.innerHTML = `
        <section aria-labelledby="results-heading" data-results-section data-query="Laptop" data-classifier="UNSPSC">
          <ul>
            <li role="listitem" data-classification-result data-code="43211507" data-name="Laptop computers" data-score="95.00"></li>
          </ul>
          <div>
            <button id="btn-csv" data-export-action="csv">Export CSV</button>
            <button id="btn-json" data-export-action="json">Export JSON</button>
          </div>
        </section>
      `;

      const createObjectURLMock = vi.fn().mockReturnValue("blob:http://localhost/csv");
      window.URL.createObjectURL = createObjectURLMock;
      window.URL.revokeObjectURL = vi.fn();

      let downloadedFilename = "";
      const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
        downloadedFilename = this.download;
      });

      const abortController = new AbortController();
      new ResultExporter(abortController.signal);

      const csvBtn = document.getElementById("btn-csv") as HTMLButtonElement;
      csvBtn.click();

      expect(createObjectURLMock).toHaveBeenCalled();
      expect(clickSpy).toHaveBeenCalled();
      expect(downloadedFilename.endsWith(".csv")).toBe(true);

      abortController.abort();
      clickSpy.mockRestore();
    });

    it("triggers JSON export when clicking [data-export-action='json']", () => {
      document.body.innerHTML = `
        <section aria-labelledby="results-heading" data-results-section data-query="Laptop" data-classifier="UNSPSC">
          <ul>
            <li role="listitem" data-classification-result data-code="43211507" data-name="Laptop computers" data-score="95.00"></li>
          </ul>
          <div>
            <button id="btn-json" data-export-action="json">Export JSON</button>
          </div>
        </section>
      `;

      const createObjectURLMock = vi.fn().mockReturnValue("blob:http://localhost/json");
      window.URL.createObjectURL = createObjectURLMock;
      window.URL.revokeObjectURL = vi.fn();

      let downloadedFilename = "";
      const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
        downloadedFilename = this.download;
      });

      const abortController = new AbortController();
      new ResultExporter(abortController.signal);

      const jsonBtn = document.getElementById("btn-json") as HTMLButtonElement;
      jsonBtn.click();

      expect(createObjectURLMock).toHaveBeenCalled();
      expect(clickSpy).toHaveBeenCalled();
      expect(downloadedFilename.endsWith(".json")).toBe(true);

      abortController.abort();
      clickSpy.mockRestore();
    });

    it("does nothing when there are no items", () => {
      document.body.innerHTML = `
        <button id="btn-csv" data-export-action="csv">Export CSV</button>
      `;

      const createObjectURLMock = vi.fn();
      window.URL.createObjectURL = createObjectURLMock;

      const abortController = new AbortController();
      new ResultExporter(abortController.signal);

      const csvBtn = document.getElementById("btn-csv") as HTMLButtonElement;
      csvBtn.click();

      expect(createObjectURLMock).not.toHaveBeenCalled();

      abortController.abort();
    });
  });
});
