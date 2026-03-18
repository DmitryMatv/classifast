import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function createScoreBarsMarkup(count = 3): string {
  return Array.from({ length: count }, (_, index) => {
    const width = Math.max(10, 100 - index * 20);
    return `<div class="score-bar" data-score-bar data-score-width="${width}"></div>`;
  }).join("");
}

function setResultsMarkup(markup: string): HTMLElement {
  const resultsContainer = document.getElementById(
    "results-container",
  ) as HTMLElement;
  resultsContainer.innerHTML = markup;
  return resultsContainer;
}

function getScoreBars(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>("[data-score-bar]"));
}

function paramsFromForm(form: HTMLFormElement): URLSearchParams {
  const formData = new FormData(form);
  const params = new URLSearchParams();
  for (const [key, value] of formData.entries()) {
    params.append(key, String(value));
  }
  return params;
}

function getInitialLoaderVals(): Record<string, unknown> {
  const loader = document.querySelector(
    "[data-initial-results-loader='true']",
  ) as HTMLElement;
  return JSON.parse(loader.getAttribute("hx-vals") ?? "{}") as Record<
    string,
    unknown
  >;
}

function sortParamEntries(params: URLSearchParams): [string, string][] {
  return Array.from(params.entries()).sort(([leftKey], [rightKey]) =>
    leftKey.localeCompare(rightKey),
  );
}

function getNavigationUrl(value: string): URL {
  return new URL(value, window.location.origin);
}

function createMediaQueryList(query: string, matches: boolean): MediaQueryList {
  return {
    matches,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(() => false),
  };
}

function createAnimationFrameController(): {
  flush: () => void;
  requestAnimationFrameMock: ReturnType<typeof vi.fn>;
} {
  const callbacks: FrameRequestCallback[] = [];
  const requestAnimationFrameMock = vi.fn((callback: FrameRequestCallback) => {
    callbacks.push(callback);
    return callbacks.length;
  });

  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    writable: true,
    value: requestAnimationFrameMock,
  });
  vi.stubGlobal("requestAnimationFrame", requestAnimationFrameMock);

  return {
    requestAnimationFrameMock,
    flush: () => {
      let iterations = 0;
      while (callbacks.length > 0) {
        const callback = callbacks.shift();
        callback?.(0);
        iterations += 1;
        if (iterations > 10) {
          throw new Error("Unexpected requestAnimationFrame loop");
        }
      }
    },
  };
}

describe("classifier.ts", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    window.history.replaceState({}, "", "/NAICS/");
    const freshBody = document.body.cloneNode(false) as HTMLBodyElement;
    document.body.replaceWith(freshBody);
    document.body.innerHTML = `
      <form
        action="/NAICS/search"
        method="get"
        hx-get="/NAICS/fragment"
        hx-push-url="false"
      >
        <input type="hidden" name="track_usage" value="true" />
        <textarea id="product_description_area" name="product_description"></textarea>
        <select id="show_top_k_categories" name="top_k">
          <option value="5">5</option>
          <option value="10" selected>10</option>
        </select>
        <select id="version_selector" name="version">
          <option value="v1" selected>v1</option>
          <option value="v2">v2</option>
        </select>
        <button type="submit">Submit</button>
      </form>
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="load"
        hx-push-url="false"
        data-initial-results-loader="true"
        hx-vals='{"version":"v1","product_description":"industrial pump","top_k":10,"track_usage":true}'
      ></div>
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

  it("animates initial score bars with a shared base delay and stagger", async () => {
    const animationFrameController = createAnimationFrameController();
    setResultsMarkup(createScoreBarsMarkup());

    await import("./classifier");
    animationFrameController.flush();

    const scoreBars = getScoreBars();
    expect(scoreBars).toHaveLength(3);
    expect(scoreBars[0]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(scoreBars[1]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(scoreBars[2]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(
      scoreBars[0]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("0ms");
    expect(
      scoreBars[1]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("100ms");
    expect(
      scoreBars[2]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("200ms");
  });

  it("auto-submits on top-k change only when textarea has content", async () => {
    await import("./classifier");
    const trigger = vi.mocked(window.htmx?.trigger);
    const navigateMock = vi.fn();
    window.__classifierNavigate = navigateMock;
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
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("uses canonical fragment params for interactive form submissions", () => {
    const form = document.querySelector("form[hx-get]") as HTMLFormElement;

    expect(sortParamEntries(paramsFromForm(form))).toEqual([
      ["product_description", ""],
      ["top_k", "10"],
      ["track_usage", "true"],
      ["version", "v1"],
    ]);
  });

  it("keeps direct-search loader params canonical without push_url", () => {
    const loaderVals = getInitialLoaderVals();

    expect(loaderVals).toEqual({
      product_description: "industrial pump",
      top_k: 10,
      track_usage: true,
      version: "v1",
    });
    expect("push_url" in loaderVals).toBe(false);
  });

  it("navigates real form submits to the server-owned search route", async () => {
    await import("./classifier");
    const navigateMock = vi.fn();
    window.__classifierNavigate = navigateMock;
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const form = document.querySelector("form[hx-get]") as HTMLFormElement;

    textarea.value = "  industrial   pump  ";
    form.dispatchEvent(
      new Event("submit", { bubbles: true, cancelable: true }),
    );

    expect(navigateMock).toHaveBeenCalledTimes(1);
    const navigationUrl = getNavigationUrl(
      navigateMock.mock.calls[0]?.[0] ?? "",
    );
    expect(navigationUrl.pathname).toBe("/NAICS/search");
    expect(navigationUrl.searchParams.get("product_description")).toBe(
      "industrial pump",
    );
    expect(navigationUrl.searchParams.get("version")).toBe("v1");
    expect(navigationUrl.searchParams.get("top_k")).toBe("10");
    expect(navigationUrl.searchParams.get("track_usage")).toBeNull();
  });

  it("normalizes whitespace and NFC before search navigation", async () => {
    await import("./classifier");
    const navigateMock = vi.fn();
    window.__classifierNavigate = navigateMock;
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const form = document.querySelector("form[hx-get]") as HTMLFormElement;

    textarea.value = "  Cafe\u0301   pump  ";
    form.dispatchEvent(
      new Event("submit", { bubbles: true, cancelable: true }),
    );

    expect(navigateMock).toHaveBeenCalledTimes(1);
    expect(
      getNavigationUrl(navigateMock.mock.calls[0]?.[0] ?? "").searchParams.get(
        "product_description",
      ),
    ).toBe("Café pump");
    expect(textarea.value).toBe("Café pump");
  });

  it("navigates on version change when there is an active query", async () => {
    await import("./classifier");
    const navigateMock = vi.fn();
    window.__classifierNavigate = navigateMock;
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const version = document.getElementById(
      "version_selector",
    ) as HTMLSelectElement;

    textarea.value = "industrial pump";
    version.value = "v2";
    version.dispatchEvent(new Event("change", { bubbles: true }));

    expect(navigateMock).toHaveBeenCalledTimes(1);
    const navigationUrl = getNavigationUrl(
      navigateMock.mock.calls[0]?.[0] ?? "",
    );
    expect(navigationUrl.pathname).toBe("/NAICS/search");
    expect(navigationUrl.searchParams.get("product_description")).toBe(
      "industrial pump",
    );
    expect(navigationUrl.searchParams.get("version")).toBe("v2");
  });

  it("keeps the initial loader history-disabled", () => {
    const loader = document.querySelector(
      "[data-initial-results-loader='true']",
    ) as HTMLElement;

    expect(loader.getAttribute("hx-push-url")).toBe("false");
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

  it("waits until HTMX afterSettle to animate swapped score bars", async () => {
    const animationFrameController = createAnimationFrameController();
    await import("./classifier");
    const resultsContainer = setResultsMarkup(createScoreBarsMarkup());
    const resultsSection = document.getElementById(
      "results-section",
    ) as HTMLElement;

    document.body.dispatchEvent(
      new CustomEvent("htmx:afterSwap", {
        detail: { target: resultsContainer },
      } as CustomEventInit),
    );

    let scoreBars = getScoreBars();
    expect(resultsSection.classList.contains("hidden")).toBe(false);
    expect(scoreBars[0]?.classList.contains("is-score-bar-visible")).toBe(
      false,
    );
    expect(
      scoreBars[0]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("");

    document.body.dispatchEvent(
      new CustomEvent("htmx:afterSettle", {
        detail: { target: resultsContainer },
      } as CustomEventInit),
    );
    animationFrameController.flush();

    scoreBars = getScoreBars();
    expect(scoreBars[0]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(scoreBars[1]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(scoreBars[2]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(
      scoreBars[0]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("0ms");
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

  it("replays score bar animation during history restore", async () => {
    const animationFrameController = createAnimationFrameController();
    await import("./classifier");
    const resultsContainer = setResultsMarkup(createScoreBarsMarkup());
    const resultsSection = document.getElementById(
      "results-section",
    ) as HTMLElement;

    document.body.dispatchEvent(new CustomEvent("htmx:historyRestore"));
    animationFrameController.flush();

    const scoreBars = getScoreBars();
    expect(resultsSection.classList.contains("hidden")).toBe(false);
    expect(scoreBars[0]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(scoreBars[1]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(scoreBars[2]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(
      scoreBars[0]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("0ms");
    expect(resultsContainer.innerHTML).toContain("score-bar");
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

  it("shows reduced-motion score bars immediately without animation delay", async () => {
    vi.mocked(window.matchMedia).mockImplementation((query: string) =>
      createMediaQueryList(query, true),
    );
    setResultsMarkup(createScoreBarsMarkup());

    await import("./classifier");

    const scoreBars = getScoreBars();
    expect(scoreBars).toHaveLength(3);
    expect(scoreBars[0]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(scoreBars[1]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(scoreBars[2]?.classList.contains("is-score-bar-visible")).toBe(true);
    expect(
      scoreBars[0]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("");
  });

  it("ignores empty results containers during HTMX swap and settle", async () => {
    await import("./classifier");
    const resultsContainer = document.getElementById(
      "results-container",
    ) as HTMLElement;
    const resultsSection = document.getElementById(
      "results-section",
    ) as HTMLElement;

    expect(resultsContainer.innerHTML).toBe("");

    document.body.dispatchEvent(
      new CustomEvent("htmx:afterSwap", {
        detail: { target: resultsContainer },
      } as CustomEventInit),
    );
    document.body.dispatchEvent(
      new CustomEvent("htmx:afterSettle", {
        detail: { target: resultsContainer },
      } as CustomEventInit),
    );

    expect(resultsContainer.innerHTML).toBe("");
    expect(resultsSection.classList.contains("hidden")).toBe(true);
    expect(getScoreBars()).toHaveLength(0);
  });
});
