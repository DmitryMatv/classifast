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
    const freshBody = document.body.cloneNode(false) as HTMLBodyElement;
    document.body.replaceWith(freshBody);
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
    ).toBe("60ms");
    expect(
      scoreBars[1]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("130ms");
    expect(
      scoreBars[2]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("200ms");
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
    ).toBe("60ms");
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
    ).toBe("60ms");
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

  it("waits for auth readiness before firing the deep-link initial loader", async () => {
    document.body.innerHTML += `
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="classifier:initial-load"
        data-initial-results-loader="true"
        data-await-auth-ready="true"
      ></div>
    `;
    delete window.Clerk;
    document.head.innerHTML = `
      <script src="https://clerk.classifast.com/npm/@clerk/clerk-js@5/dist/clerk.browser.js"></script>
    `;

    await import("./classifier");

    expect(window.htmx?.process).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).not.toHaveBeenCalled();

    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));

    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      "classifier:initial-load",
    );
  });

  it("falls back to firing the deferred deep-link loader after the auth timeout", async () => {
    document.body.innerHTML += `
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="classifier:initial-load"
        data-initial-results-loader="true"
        data-await-auth-ready="true"
      ></div>
    `;
    delete window.Clerk;
    document.head.innerHTML = `
      <script src="https://clerk.classifast.com/npm/@clerk/clerk-js@5/dist/clerk.browser.js"></script>
    `;

    await import("./classifier");

    expect(window.htmx?.trigger).not.toHaveBeenCalled();

    vi.advanceTimersByTime(3999);
    expect(window.htmx?.trigger).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      "classifier:initial-load",
    );
  });

  it("cancels the auth timeout when auth becomes ready first", async () => {
    document.body.innerHTML += `
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="classifier:initial-load"
        data-initial-results-loader="true"
        data-await-auth-ready="true"
      ></div>
    `;
    delete window.Clerk;
    document.head.innerHTML = `
      <script src="https://clerk.classifast.com/npm/@clerk/clerk-js@5/dist/clerk.browser.js"></script>
    `;

    await import("./classifier");

    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(4000);
    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
  });

  it("cancels the deferred deep-link loader after a manual results request starts", async () => {
    document.body.innerHTML += `
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="classifier:initial-load"
        data-initial-results-loader="true"
        data-await-auth-ready="true"
      ></div>
    `;
    delete window.Clerk;
    document.head.innerHTML = `
      <script src="https://clerk.classifast.com/npm/@clerk/clerk-js@5/dist/clerk.browser.js"></script>
    `;

    await import("./classifier");

    const form = document.querySelector("form") as HTMLFormElement;
    const resultsContainer = document.getElementById(
      "results-container",
    ) as HTMLElement;

    document.body.dispatchEvent(
      new CustomEvent("htmx:beforeRequest", {
        detail: { elt: form, target: resultsContainer },
      } as CustomEventInit),
    );

    expect(
      document.querySelector("[data-initial-results-loader='true']"),
    ).toBeNull();

    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    vi.advanceTimersByTime(4000);

    expect(window.htmx?.trigger).not.toHaveBeenCalled();
  });

  it("does not cancel the deferred loader when the loader itself starts the request", async () => {
    document.body.innerHTML += `
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="classifier:initial-load"
        data-initial-results-loader="true"
        data-await-auth-ready="true"
      ></div>
    `;
    delete window.Clerk;
    document.head.innerHTML = `
      <script src="https://clerk.classifast.com/npm/@clerk/clerk-js@5/dist/clerk.browser.js"></script>
    `;

    await import("./classifier");

    const loader = document.querySelector(
      "[data-initial-results-loader='true']",
    ) as HTMLElement;
    const resultsContainer = document.getElementById(
      "results-container",
    ) as HTMLElement;

    document.body.dispatchEvent(
      new CustomEvent("htmx:beforeRequest", {
        detail: { elt: loader, target: resultsContainer },
      } as CustomEventInit),
    );
    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));

    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).toHaveBeenCalledWith(
      loader,
      "classifier:initial-load",
    );
  });

  it("fires the deep-link initial loader immediately when auth is already ready", async () => {
    document.body.innerHTML += `
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="classifier:initial-load"
        data-initial-results-loader="true"
        data-await-auth-ready="true"
      ></div>
    `;
    window.__clerkAuthReady = true;

    await import("./classifier");

    expect(window.htmx?.process).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      "classifier:initial-load",
    );
  });

  it("fires example-query initial loaders immediately without waiting for auth", async () => {
    document.body.innerHTML += `
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="classifier:initial-load"
        data-initial-results-loader="true"
        data-await-auth-ready="false"
      ></div>
    `;

    await import("./classifier");

    expect(window.htmx?.process).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      "classifier:initial-load",
    );
  });

  it("does not fire the initial loader twice when auth was already ready", async () => {
    document.body.innerHTML += `
      <div
        hx-get="/NAICS/fragment"
        hx-trigger="classifier:initial-load"
        data-initial-results-loader="true"
        data-await-auth-ready="true"
      ></div>
    `;
    window.__clerkAuthReady = true;

    await import("./classifier");
    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));

    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
  });
});
