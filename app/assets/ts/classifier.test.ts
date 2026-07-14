import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function createScoreBarsMarkup(count = 3): string {
  return Array.from({ length: count }, (_, index) => {
    const width = Math.max(10, 100 - index * 20);
    return `<div class="score-bar" data-score-bar data-score-width="${width}"></div>`;
  }).join("");
}

async function flushAsyncWork(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
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

function getClassifierForm(): HTMLFormElement {
  return document.getElementById("classifier-form") as HTMLFormElement;
}

function createConfigRequestDetail(
  form: HTMLFormElement,
): HtmxConfigRequestEvent["detail"] {
  return {
    headers: {},
    xhr: {} as XMLHttpRequest,
    elt: form,
    parameters: {
      product_description: (
        document.getElementById(
          "product_description_area",
        ) as HTMLTextAreaElement
      ).value,
      top_k: (
        document.getElementById("show_top_k_categories") as HTMLSelectElement
      ).value,
      version: (
        document.getElementById("version_selector") as HTMLSelectElement
      ).value,
    } as Record<string, unknown>,
  };
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

function createAnimationFrameController() {
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
      <form
        id="classifier-form"
        hx-get="/NAICS/fragment"
        hx-sync="this:drop"
        data-default-example-prefill="false"
        data-initial-query-present="false"
        data-autoload-enabled="false"
        data-default-top-k="10"
        data-default-version="v1"
      >
        <textarea id="product_description_area" name="product_description"></textarea>
        <select id="show_top_k_categories" name="top_k">
          <option value="5">5</option>
          <option value="10" selected>10</option>
          <option value="30">30</option>
        </select>
        <select id="version_selector" name="version">
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
    ).toBe("0ms");
    expect(
      scoreBars[1]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("0ms");
    expect(
      scoreBars[2]?.style.getPropertyValue("--score-animation-delay"),
    ).toBe("0ms");
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

  it("waits for auth readiness before sending a manual form submission", async () => {
    window.__authReady = false;
    await import("./classifier");
    const form = getClassifierForm();
    const firstSubmit = new Event("submit", {
      bubbles: true,
      cancelable: true,
    });
    const duplicateSubmit = new Event("submit", {
      bubbles: true,
      cancelable: true,
    });

    form.dispatchEvent(firstSubmit);
    form.dispatchEvent(duplicateSubmit);

    expect(firstSubmit.defaultPrevented).toBe(true);
    expect(duplicateSubmit.defaultPrevented).toBe(true);
    expect(window.htmx?.trigger).not.toHaveBeenCalledWith(form, "submit");

    window.__authReady = true;
    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));

    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
    expect(window.htmx?.trigger).toHaveBeenCalledWith(form, "submit");
  });

  it("keeps default top-k and omits default version on manual requests", async () => {
    await import("./classifier");
    const form = getClassifierForm();
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "industrial pump";

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["product_description"]).toBe("industrial pump");
    expect(detail.parameters["top_k"]).toBe("10");
    expect(detail.parameters["version"]).toBeUndefined();
    expect(detail.parameters["push_url"]).toBeUndefined();
    expect(detail.parameters["track_usage"]).toBeUndefined();
  });

  it("keeps non-default top-k on manual requests", async () => {
    await import("./classifier");
    const form = getClassifierForm();
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const topK = document.getElementById(
      "show_top_k_categories",
    ) as HTMLSelectElement;
    textarea.value = "industrial pump";
    topK.value = "30";

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["top_k"]).toBe("30");
    expect(detail.parameters["version"]).toBeUndefined();
  });

  it("keeps non-default version on manual requests", async () => {
    await import("./classifier");
    const form = getClassifierForm();
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const versionSelector = document.getElementById(
      "version_selector",
    ) as HTMLSelectElement;
    textarea.value = "industrial pump";
    versionSelector.value = "v2";

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["top_k"]).toBe("10");
    expect(detail.parameters["version"]).toBe("v2");
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

  it("hides the description block when description text is empty", async () => {
    const toggle = document.getElementById(
      "description-toggle",
    ) as HTMLButtonElement;
    const container = document.getElementById(
      "description-container",
    ) as HTMLElement;
    const content = document.getElementById(
      "description-content",
    ) as HTMLElement;
    content.textContent = "   ";

    await import("./classifier");

    expect(toggle.style.display).toBe("none");
    expect(container.style.display).toBe("none");
  });

  it("applies initially expanded description state", async () => {
    const toggle = document.getElementById(
      "description-toggle",
    ) as HTMLButtonElement;
    const content = document.getElementById(
      "description-content",
    ) as HTMLElement;
    const logo = document.querySelector(
      '[data-classifier-logo="true"]',
    ) as HTMLElement;
    toggle.setAttribute("aria-expanded", "true");

    await import("./classifier");

    expect(toggle.textContent).toBe("Show less");
    expect(content.style.display).toBe("block");
    expect(content.getAttribute("aria-hidden")).toBe("false");
    expect(logo.style.display).toBe("none");
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

  it("autoloads auth-gated initial results when auth is already ready", async () => {
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    window.__authReady = true;

    await import("./classifier");
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).toHaveBeenCalledWith(form, "submit");

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["push_url"]).toBeUndefined();
    expect(detail.parameters["track_usage"]).toBeUndefined();
    expect(detail.parameters["top_k"]).toBe("10");
    expect(detail.parameters["version"]).toBeUndefined();
  });

  it("waits for authReady before autoloading auth-gated initial results", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";

    await import("./classifier");
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).not.toHaveBeenCalledWith(form, "submit");

    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).toHaveBeenCalledWith(form, "submit");
  });

  it("suppresses deep-link autoload history pushes without changing the request URL", async () => {
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    window.__authReady = true;

    window.history.replaceState({}, "", "/NAICS/industrial_pump?version=v2");

    await import("./classifier");
    vi.advanceTimersByTime(0);

    const historyDetail = {
      history: {
        type: "push",
        path: "/NAICS/industrial_pump?version=v2",
      },
    };
    document.body.dispatchEvent(
      new CustomEvent("htmx:beforeHistoryUpdate", {
        detail: historyDetail,
      } as CustomEventInit),
    );

    expect(historyDetail.history.type).toBe("replace");
    expect(historyDetail.history.path).toBe(
      "/NAICS/industrial_pump?version=v2",
    );
  });

  it("suppresses history changes for base example autoload", async () => {
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["defaultExamplePrefill"] = "true";
    form.dataset["initialQueryPresent"] = "false";
    window.__authReady = true;
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "Industrial pump";
    window.history.replaceState({}, "", "/NAICS/");

    await import("./classifier");
    vi.advanceTimersByTime(0);

    const historyDetail = {
      history: {
        type: "push",
        path: "/NAICS/industrial_pump",
      },
    };
    document.body.dispatchEvent(
      new CustomEvent("htmx:beforeHistoryUpdate", {
        detail: historyDetail,
      } as CustomEventInit),
    );

    expect(historyDetail.history.type).toBe("replace");
    expect(historyDetail.history.path).toBe("/NAICS/");
  });

  it("cancels auth-gated autoload when a manual results request starts before authReady", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";

    await import("./classifier");

    document.body.dispatchEvent(
      new CustomEvent("htmx:beforeRequest", {
        detail: {
          target: document.getElementById("results-container"),
          elt: form,
        },
      } as CustomEventInit),
    );
    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).not.toHaveBeenCalledWith(form, "submit");
  });

  it("deep-link autoload only submits once even if authReady fires again later", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";

    await import("./classifier");

    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    vi.advanceTimersByTime(0);
    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);

    document.body.dispatchEvent(
      new CustomEvent("htmx:afterRequest", {
        detail: {
          target: document.getElementById("results-container"),
          elt: form,
        },
      } as CustomEventInit),
    );
    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).toHaveBeenCalledTimes(1);
  });

  it("waits for auth before autoloading the base example query", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "false";
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    form.dataset["defaultExamplePrefill"] = "true";
    textarea.value = "Industrial pump";

    await import("./classifier");
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).not.toHaveBeenCalledWith(form, "submit");

    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).toHaveBeenCalledWith(form, "submit");
    expect(textarea.value).toBe("Industrial pump");

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["push_url"]).toBeUndefined();
    expect(detail.parameters["track_usage"]).toBeUndefined();
    expect(detail.parameters["product_description"]).toBe("Industrial pump");
    expect(detail.parameters["top_k"]).toBe("10");
    expect(detail.parameters["version"]).toBeUndefined();
  });

  it("keeps the default example query active for top-k changes after timer clear", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["defaultExamplePrefill"] = "true";
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "false";
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const topK = document.getElementById(
      "show_top_k_categories",
    ) as HTMLSelectElement;
    textarea.value = "Industrial pump";

    await import("./classifier");
    vi.advanceTimersByTime(0);
    vi.mocked(window.htmx!.trigger).mockClear();

    textarea.value = "";
    topK.dispatchEvent(new Event("change", { bubbles: true }));

    expect(window.htmx?.trigger).toHaveBeenCalledWith(form, "submit");
  });

  it("uses the stored default example query when configRequest runs after timer clear", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["defaultExamplePrefill"] = "true";
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "false";
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "Industrial pump";

    await import("./classifier");
    vi.advanceTimersByTime(0);

    textarea.value = "";

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["product_description"]).toBe("Industrial pump");
    expect(detail.parameters["push_url"]).toBeUndefined();
    expect(detail.parameters["track_usage"]).toBeUndefined();
    expect(detail.parameters["top_k"]).toBe("10");
    expect(detail.parameters["version"]).toBeUndefined();
  });

  it("replaces the stored default example query after real user input", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["defaultExamplePrefill"] = "true";
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "false";
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "Industrial pump";

    await import("./classifier");

    textarea.value = "custom typed query";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["product_description"]).toBe("custom typed query");
  });

  it("drops the hidden fallback query after the user clears the textarea", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["defaultExamplePrefill"] = "true";
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "false";
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const topK = document.getElementById(
      "show_top_k_categories",
    ) as HTMLSelectElement;
    textarea.value = "Industrial pump";

    await import("./classifier");
    vi.advanceTimersByTime(0);
    vi.mocked(window.htmx!.trigger).mockClear();

    textarea.value = "";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    topK.dispatchEvent(new Event("change", { bubbles: true }));

    expect(window.htmx?.trigger).not.toHaveBeenCalled();

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["product_description"]).toBe("");
  });

  it("does not create a hidden fallback query for deep-link prefills", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["defaultExamplePrefill"] = "false";
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "helicopter taxi";

    await import("./classifier");

    textarea.value = "";

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["product_description"]).toBe("");
  });

  it("preserves history metadata for the hidden active query without repopulating the textarea", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["defaultExamplePrefill"] = "true";
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "false";
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "Industrial pump";

    await import("./classifier");

    textarea.value = "";
    document.body.dispatchEvent(new CustomEvent("htmx:beforeHistorySave"));

    expect(textarea.value).toBe("");
    expect(textarea.defaultValue).toBe("Industrial pump");
    expect(textarea.textContent).toBe("Industrial pump");
  });

  it("keeps the deep-link query in the textarea during auth-gated autoload", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "helicopter taxi";

    await import("./classifier");
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).toHaveBeenCalledWith(form, "submit");
    expect(textarea.value).toBe("helicopter taxi");
  });

  it("clears one-shot autoload overrides immediately after configRequest", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";

    await import("./classifier");
    vi.advanceTimersByTime(0);

    const firstDetail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail: firstDetail,
      } as CustomEventInit),
    );
    expect(firstDetail.parameters["push_url"]).toBeUndefined();
    expect(firstDetail.parameters["track_usage"]).toBeUndefined();

    const secondDetail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail: secondDetail,
      } as CustomEventInit),
    );

    expect(secondDetail.parameters["push_url"]).toBeUndefined();
    expect(secondDetail.parameters["track_usage"]).toBeUndefined();
  });

  it("completes autoload state on 429 responses from the form request", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    const resultsContainer = document.getElementById(
      "results-container",
    ) as HTMLElement;

    await import("./classifier");
    vi.advanceTimersByTime(0);

    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail: createConfigRequestDetail(form),
      } as CustomEventInit),
    );
    document.body.dispatchEvent(
      new CustomEvent("htmx:responseError", {
        detail: {
          xhr: { status: 429, response: "<div>Paywall</div>" },
          target: resultsContainer,
          elt: form,
        },
      } as CustomEventInit),
    );

    expect(resultsContainer.innerHTML).toContain("Paywall");

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["push_url"]).toBeUndefined();
    expect(detail.parameters["track_usage"]).toBeUndefined();
  });

  it("swaps quota-unavailable 503 responses into the results container", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    const resultsContainer = document.getElementById(
      "results-container",
    ) as HTMLElement;

    await import("./classifier");
    vi.advanceTimersByTime(0);

    document.body.dispatchEvent(
      new CustomEvent("htmx:responseError", {
        detail: {
          xhr: {
            status: 503,
            response: "<div>Usage tracking is temporarily unavailable</div>",
          },
          target: resultsContainer,
          elt: form,
        },
      } as CustomEventInit),
    );

    expect(resultsContainer.innerHTML).toContain(
      "Usage tracking is temporarily unavailable",
    );
  });

  it("clears autoload override state after request completion", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";

    await import("./classifier");
    vi.advanceTimersByTime(0);

    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail: createConfigRequestDetail(form),
      } as CustomEventInit),
    );
    document.body.dispatchEvent(
      new CustomEvent("htmx:afterRequest", {
        detail: {
          target: document.getElementById("results-container"),
          elt: form,
        },
      } as CustomEventInit),
    );

    const detail = createConfigRequestDetail(form);
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        detail,
      } as CustomEventInit),
    );

    expect(detail.parameters["push_url"]).toBeUndefined();
    expect(detail.parameters["track_usage"]).toBeUndefined();
  });

  it("does not retrigger autoload during history restore", async () => {
    window.__authReady = true;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";

    await import("./classifier");
    vi.advanceTimersByTime(0);
    vi.mocked(window.htmx!.trigger).mockClear();

    document.body.dispatchEvent(new CustomEvent("htmx:historyRestore"));

    expect(window.htmx?.trigger).not.toHaveBeenCalled();
  });

  it("cancels pending autoload when the user edits the textarea before authReady", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";

    await import("./classifier");

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "edited query";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));

    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    vi.advanceTimersByTime(0);

    expect(window.htmx?.trigger).not.toHaveBeenCalledWith(form, "submit");
  });

  it("shows the loading indicator while waiting for authReady", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    const indicator = document.getElementById(
      "loading-indicator",
    ) as HTMLElement;

    await import("./classifier");

    expect(indicator.classList.contains("htmx-request")).toBe(true);
    expect(window.htmx?.trigger).not.toHaveBeenCalled();
  });

  it("autoloads auth-gated initial results after a delayed authReady fallback signal", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    window.setTimeout(() => {
      window.__authReady = true;
      document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    }, 4000);

    await import("./classifier");

    expect(window.htmx?.trigger).not.toHaveBeenCalled();

    vi.advanceTimersByTime(4000);
    vi.runOnlyPendingTimers();
    await flushAsyncWork();

    expect(window.__authReady).toBe(true);
    expect(window.htmx?.trigger).toHaveBeenCalledWith(form, "submit");
  });

  it("manual request path keeps spinner behavior consistent", async () => {
    window.__authReady = false;
    vi.doMock("./common", () => ({
      ShareLink: {
        copyShareableLink: vi.fn(),
      },
    }));
    const form = getClassifierForm();
    form.dataset["autoloadEnabled"] = "true";
    form.dataset["initialQueryPresent"] = "true";
    const indicator = document.getElementById(
      "loading-indicator",
    ) as HTMLElement;

    await import("./classifier");
    indicator.classList.remove("htmx-request"); // Clear initial state

    document.body.dispatchEvent(
      new CustomEvent("htmx:beforeRequest", {
        detail: {
          target: document.getElementById("results-container"),
          elt: form,
        },
      } as CustomEventInit),
    );
    expect(indicator.classList.contains("htmx-request")).toBe(true);

    document.body.dispatchEvent(
      new CustomEvent("htmx:afterRequest", {
        detail: {
          target: document.getElementById("results-container"),
          elt: form,
        },
      } as CustomEventInit),
    );
    expect(indicator.classList.contains("htmx-request")).toBe(false);
  });
});
