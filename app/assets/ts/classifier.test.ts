import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("classifier.ts", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    document.body.innerHTML = `
      <form hx-get="/NAICS/fragment">
        <textarea id="product_description_area"></textarea>
        <select id="show_top_k_categories">
          <option value="5">5</option>
          <option value="10" selected>10</option>
        </select>
        <select id="version_selector">
          <option value="v1" selected>v1</option>
          <option value="v2">v2</option>
        </select>
        <button type="submit">Submit</button>
      </form>
      <div id="loading-indicator"></div>
      <section id="results-section" class="hidden">
        <div id="results-container"></div>
      </section>
      <button id="description-toggle" aria-expanded="false" data-classifier-type="NAICS"></button>
      <div id="description-container">
        <div id="description-content" aria-hidden="true">Useful description</div>
      </div>
      <div data-classifier-logo="true"></div>
      <button id="share-button">Share</button>
    `;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("auto-submits on top-k change only when textarea has content", async () => {
    await import("./classifier");
    const trigger = vi.mocked(window.htmx?.trigger);
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const topK = document.getElementById(
      "show_top_k_categories",
    ) as HTMLSelectElement;

    topK.dispatchEvent(new Event("change", { bubbles: true }));
    expect(trigger).not.toHaveBeenCalled();

    textarea.value = "industrial pump";
    topK.dispatchEvent(new Event("change", { bubbles: true }));
    expect(trigger).toHaveBeenCalled();
  });

  it("shows and hides the loading indicator across HTMX request events", async () => {
    await import("./classifier");
    const indicator = document.getElementById(
      "loading-indicator",
    ) as HTMLElement;

    document.body.dispatchEvent(
      new CustomEvent("htmx:beforeRequest", {
        detail: { target: document.getElementById("results-container") },
      } as CustomEventInit),
    );
    expect(indicator.classList.contains("htmx-request")).toBe(true);

    document.body.dispatchEvent(
      new CustomEvent("htmx:afterRequest", {
        detail: {
          target: document.getElementById("results-container"),
          elt: document.createElement("div"),
        },
      } as CustomEventInit),
    );
    expect(indicator.classList.contains("htmx-request")).toBe(false);
  });

  it("injects rate limit responses into the results container", async () => {
    await import("./classifier");
    const resultsContainer = document.getElementById(
      "results-container",
    ) as HTMLElement;
    const resultsSection = document.getElementById(
      "results-section",
    ) as HTMLElement;

    document.body.dispatchEvent(
      new CustomEvent("htmx:responseError", {
        detail: {
          xhr: { status: 429, response: "<div>Paywall</div>" },
          target: resultsContainer,
        },
      } as CustomEventInit),
    );

    expect(resultsContainer.innerHTML).toContain("Paywall");
    expect(resultsSection.classList.contains("hidden")).toBe(false);
  });

  it("toggles description content and updates labels", async () => {
    await import("./classifier");
    const toggle = document.getElementById(
      "description-toggle",
    ) as HTMLButtonElement;
    const content = document.getElementById(
      "description-content",
    ) as HTMLElement;
    const logo = document.querySelector(
      '[data-classifier-logo="true"]',
    ) as HTMLElement;

    expect(toggle.textContent).toBe("Learn more about NAICS");
    expect(content.style.display).toBe("none");

    toggle.click();
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(toggle.textContent).toBe("Show less");
    expect(content.style.display).toBe("block");
    expect(logo.style.display).toBe("none");

    toggle.click();
    expect(toggle.textContent).toBe("Learn more about NAICS");
    expect(content.style.display).toBe("none");
  });

  it("syncs form state before history save and re-runs results visibility on restore", async () => {
    await import("./classifier");
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const version = document.getElementById(
      "version_selector",
    ) as HTMLSelectElement;
    const topK = document.getElementById(
      "show_top_k_categories",
    ) as HTMLSelectElement;
    const resultsContainer = document.getElementById(
      "results-container",
    ) as HTMLElement;
    const resultsSection = document.getElementById(
      "results-section",
    ) as HTMLElement;

    textarea.value = "updated query";
    version.selectedIndex = 1;
    Array.from(version.options).forEach((option, index) => {
      option.selected = index === 1;
    });
    topK.selectedIndex = 0;
    Array.from(topK.options).forEach((option, index) => {
      option.selected = index === 0;
    });

    document.body.dispatchEvent(new CustomEvent("htmx:beforeHistorySave"));

    expect(textarea.defaultValue).toBe("updated query");
    expect(version.options[1]?.defaultSelected).toBe(true);
    expect(topK.options[0]?.defaultSelected).toBe(true);

    resultsContainer.innerHTML = "<div>Results</div>";
    document.body.dispatchEvent(new CustomEvent("htmx:historyRestore"));

    expect(resultsSection.classList.contains("hidden")).toBe(false);
  });
});
