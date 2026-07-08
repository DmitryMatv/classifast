import { ShareLink } from "./common";

const BASE_SCORE_BAR_DELAY_MS = 0;
const SCORE_BAR_STAGGER_MS = 0; // 100 before
const MAX_SCORE_BAR_STAGGER_MS = 0; // 1000 before

type DescriptionToggleElements = {
  toggleButton: HTMLButtonElement;
  descriptionContent: HTMLElement;
  container: HTMLElement;
};

/**
 * Classifier page specific functionality
 * Handles form auto-submission, HTMX event handling, and UI interactions
 */

class ClassifierPage {
  private autoloadStatus:
    | "idle"
    | "pending"
    | "triggered"
    | "cancelled"
    | "completed" = "idle";
  private pendingAutoloadRequestConfig: {
    trackUsage: boolean;
    suppressUrlChange: boolean;
  } | null = null;
  private autoloadRequestInFlight = false;
  private suppressNextHistoryUpdate = false;
  private activeQuery: string | null = null;
  private defaultExampleQuery: string | null = null;

  constructor() {
    this.init();
  }

  private init(): void {
    document.documentElement.classList.add("js-score-animations");
    this.initializeQueryState();
    this.setupInitialResultsAutoload();
    this.setupTopKAutosubmit();
    this.setupHTMXListeners();
    this.setupDescriptionToggle();
    this.setupAutoloadCancellationListeners();
    this.setupQueryStateTracking();
    this.attachShareButtonListener();
    this.animateScoreBars(document);
  }

  private getLoadingIndicator(): HTMLElement | null {
    return document.getElementById("loading-indicator");
  }

  private showLoadingIndicator(): void {
    this.getLoadingIndicator()?.classList.add("htmx-request");
  }

  private hideLoadingIndicator(): void {
    this.getLoadingIndicator()?.classList.remove("htmx-request");
  }

  private isResultsTarget(target: EventTarget | null): target is HTMLElement {
    return target instanceof HTMLElement && target.id === "results-container";
  }

  private getClassifierForm(): HTMLFormElement | null {
    return document.getElementById("classifier-form") as HTMLFormElement | null;
  }

  private getAutoloadConfig(): {
    enabled: boolean;
    requiresAuthReady: boolean;
    trackUsage: boolean;
    suppressUrlChange: boolean;
  } | null {
    const form = this.getClassifierForm();
    if (!form) {
      return null;
    }

    return {
      enabled: form.dataset["autoloadEnabled"] === "true",
      requiresAuthReady: form.dataset["initialQueryPresent"] === "true",
      trackUsage: form.dataset["initialTrackUsage"] === "true",
      suppressUrlChange: true,
    };
  }

  private getDefaultTopK(): string | null {
    return this.getClassifierForm()?.dataset["defaultTopK"] ?? null;
  }

  private getDefaultVersion(): string | null {
    return this.getClassifierForm()?.dataset["defaultVersion"] ?? null;
  }

  private getTopKSelector(): HTMLSelectElement | null {
    return document.getElementById(
      "show_top_k_categories",
    ) as HTMLSelectElement | null;
  }

  private getVersionSelector(): HTMLSelectElement | null {
    return document.getElementById(
      "version_selector",
    ) as HTMLSelectElement | null;
  }

  private canonicalizeDefaultParameters(
    parameters: Record<string, unknown>,
  ): void {
    const topKSelector = this.getTopKSelector();
    if (topKSelector) {
      parameters["top_k"] = topKSelector.value;
    }

    const versionSelector = this.getVersionSelector();
    const defaultVersion = this.getDefaultVersion();
    if (versionSelector) {
      if (defaultVersion && versionSelector.value === defaultVersion) {
        delete parameters["version"];
      } else {
        parameters["version"] = versionSelector.value;
      }
    }
  }

  private getProductDescriptionArea(): HTMLTextAreaElement | null {
    return document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement | null;
  }

  private initializeQueryState(): void {
    const form = this.getClassifierForm();
    const productDescriptionArea = this.getProductDescriptionArea();

    if (!form || !productDescriptionArea) {
      return;
    }

    if (
      form.dataset["defaultExamplePrefill"] === "true" &&
      form.dataset["initialQueryPresent"] !== "true" &&
      productDescriptionArea.value.trim()
    ) {
      this.defaultExampleQuery = productDescriptionArea.value;
      this.activeQuery = productDescriptionArea.value;
    }
  }

  private getEffectiveQuery(): string {
    const productDescriptionArea = this.getProductDescriptionArea();
    const textareaValue = productDescriptionArea?.value.trim() ?? "";
    if (textareaValue) {
      return productDescriptionArea?.value ?? "";
    }

    return this.activeQuery ?? this.defaultExampleQuery ?? "";
  }

  private setupQueryStateTracking(): void {
    const productDescriptionArea = this.getProductDescriptionArea();

    if (!productDescriptionArea) {
      return;
    }

    productDescriptionArea.addEventListener("input", () => {
      if (productDescriptionArea.value.trim()) {
        this.activeQuery = productDescriptionArea.value;
      } else {
        this.activeQuery = null;
      }

      this.defaultExampleQuery = null;
    });
  }

  private isAutoloadRequest(element: Element | null): boolean {
    const form = this.getClassifierForm();
    return !!(form && element === form && this.autoloadRequestInFlight);
  }

  private cancelInitialResultsAutoload(): void {
    if (
      this.autoloadStatus === "completed" ||
      this.autoloadStatus === "triggered"
    ) {
      return;
    }

    this.autoloadStatus = "cancelled";
    this.hideLoadingIndicator();
    this.pendingAutoloadRequestConfig = null;
    this.autoloadRequestInFlight = false;
    this.suppressNextHistoryUpdate = false;
  }

  private completeInitialResultsAutoload(): void {
    this.autoloadStatus = "completed";
    this.pendingAutoloadRequestConfig = null;
    this.autoloadRequestInFlight = false;
    this.suppressNextHistoryUpdate = false;
  }

  private clearAutoloadRequestState(): void {
    this.pendingAutoloadRequestConfig = null;
    this.autoloadRequestInFlight = false;
    this.suppressNextHistoryUpdate = false;
  }

  private submitInitialResultsAutoload(
    trackUsage: boolean,
    suppressUrlChange: boolean,
  ): void {
    const form = this.getClassifierForm();

    if (
      !form ||
      this.autoloadStatus === "triggered" ||
      this.autoloadStatus === "completed" ||
      this.autoloadStatus === "cancelled" ||
      !window.htmx
    ) {
      return;
    }

    this.autoloadStatus = "triggered";
    this.pendingAutoloadRequestConfig = { trackUsage, suppressUrlChange };
    this.autoloadRequestInFlight = true;
    this.suppressNextHistoryUpdate = trackUsage && suppressUrlChange;
    window.htmx.trigger(form, "submit");
  }

  private setupInitialResultsAutoload(): void {
    const config = this.getAutoloadConfig();
    if (!config?.enabled) {
      return;
    }

    const triggerInitialResultsLoad = () => {
      if (
        this.autoloadStatus === "cancelled" ||
        this.autoloadStatus === "triggered" ||
        this.autoloadStatus === "completed"
      ) {
        return;
      }

      this.submitInitialResultsAutoload(
        config.trackUsage,
        config.suppressUrlChange,
      );
    };

    const scheduleInitialResultsLoad = () => {
      if (this.autoloadStatus === "cancelled") {
        return;
      }

      window.setTimeout(triggerInitialResultsLoad, 0);
    };

    this.autoloadStatus = "pending";

    if (!config.requiresAuthReady) {
      scheduleInitialResultsLoad();
      return;
    }

    if (window.__authReady) {
      scheduleInitialResultsLoad();
      return;
    }

    this.showLoadingIndicator();

    const authTimeout = window.setTimeout(() => {
      this.hideLoadingIndicator();
    }, 10000); // 10 second fallback

    document.body.addEventListener(
      "htmx:authReady",
      () => {
        window.clearTimeout(authTimeout);
        scheduleInitialResultsLoad();
      },
      { once: true },
    );
  }

  private ensureResultsSectionVisible(): void {
    const resultsContainer = document.getElementById("results-container");
    const resultsSection = document.getElementById("results-section");

    if (!resultsContainer || !resultsSection) {
      return;
    }

    if (resultsContainer.innerHTML.trim()) {
      resultsSection.classList.remove("hidden");
    }
  }

  private syncTextareaState(): void {
    const productDescriptionArea = this.getProductDescriptionArea();

    if (!productDescriptionArea) {
      return;
    }

    const syncedValue =
      productDescriptionArea.value ||
      this.activeQuery ||
      this.defaultExampleQuery ||
      "";
    productDescriptionArea.defaultValue = syncedValue;
    productDescriptionArea.textContent = syncedValue;
  }

  private syncSelectState(selectId: string): void {
    const select = document.getElementById(
      selectId,
    ) as HTMLSelectElement | null;

    if (!select) {
      return;
    }

    Array.from(select.options).forEach((option) => {
      const isSelected = option.selected;
      option.defaultSelected = isSelected;
      option.toggleAttribute("selected", isSelected);
    });
  }

  private syncHistoryState(): void {
    this.syncTextareaState();
    this.syncSelectState("version_selector");
    this.syncSelectState("show_top_k_categories");
    this.hideLoadingIndicator();
  }

  private animateScoreBars(root: ParentNode = document): void {
    const scoreBars = Array.from(
      root.querySelectorAll<HTMLElement>("[data-score-bar]"),
    );

    if (scoreBars.length === 0) {
      return;
    }

    scoreBars.forEach((bar) => {
      const rawScoreWidth = Number(bar.dataset["scoreWidth"] ?? "0");
      const scoreWidth = Number.isFinite(rawScoreWidth)
        ? Math.min(Math.max(rawScoreWidth, 0), 100)
        : 0;
      bar.style.width = `${scoreWidth}%`;
    });

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (prefersReducedMotion) {
      scoreBars.forEach((bar) => {
        bar.classList.add("is-score-bar-visible");
      });
      return;
    }

    scoreBars.forEach((bar, index) => {
      bar.classList.remove("is-score-bar-visible");
      bar.style.setProperty(
        "--score-animation-delay",
        `${BASE_SCORE_BAR_DELAY_MS + Math.min(index * SCORE_BAR_STAGGER_MS, MAX_SCORE_BAR_STAGGER_MS)}ms`,
      );
    });

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        scoreBars.forEach((bar) => {
          bar.classList.add("is-score-bar-visible");
        });
      });
    });
  }

  private handleResultsSwap(): void {
    this.ensureResultsSectionVisible();
    this.attachShareButtonListener();
  }

  private handleResultsSettle(): void {
    const resultsContainer = document.getElementById("results-container");
    if (resultsContainer) {
      this.animateScoreBars(resultsContainer);
    }
  }

  /**
   * Setup automatic form submission when Top K selector changes
   * Only submits if there's text in the description area
   */
  private setupTopKAutosubmit(): void {
    const topKSelector = document.getElementById(
      "show_top_k_categories",
    ) as HTMLSelectElement | null;
    const productDescriptionArea = this.getProductDescriptionArea();

    if (topKSelector && productDescriptionArea) {
      topKSelector.addEventListener("change", () => {
        if (this.getEffectiveQuery()) {
          this.triggerFormSubmission();
        }
      });
    }
  }

  /**
   * Trigger form submission with visual feedback
   */
  private triggerFormSubmission(): void {
    const form = document.querySelector(
      "form[hx-get]",
    ) as HTMLFormElement | null;
    const submitBtn = form?.querySelector(
      'button[type="submit"]',
    ) as HTMLElement | null;

    if (form) {
      if (submitBtn) {
        submitBtn.classList.add("active", "scale-95");
        setTimeout(() => {
          submitBtn.classList.remove("active", "scale-95");
        }, 150);
      }
      // Use HTMX to trigger the form submission
      if (window.htmx) {
        window.htmx.trigger(form, "submit");
      }
    }
  }

  private setupAutoloadCancellationListeners(): void {
    const productDescriptionArea = this.getProductDescriptionArea();

    if (!productDescriptionArea) {
      return;
    }

    productDescriptionArea.addEventListener("input", () => {
      if (this.autoloadStatus === "pending") {
        this.cancelInitialResultsAutoload();
      }
    });
  }

  /**
   * Setup HTMX event listeners for response handling
   */
  private setupHTMXListeners(): void {
    document.body.addEventListener("htmx:configRequest", (evt: Event) => {
      const htmxEvent = evt as HtmxConfigRequestEvent;
      const form = this.getClassifierForm();

      if (!form || htmxEvent.detail.elt !== form) {
        return;
      }

      const effectiveQuery = this.getEffectiveQuery();
      if (effectiveQuery) {
        htmxEvent.detail.parameters["product_description"] = effectiveQuery;
      }
      this.canonicalizeDefaultParameters(htmxEvent.detail.parameters);

      if (!this.pendingAutoloadRequestConfig) {
        return;
      }

      if (!this.pendingAutoloadRequestConfig.trackUsage) {
        htmxEvent.detail.parameters["track_usage"] = "false";
      } else {
        delete htmxEvent.detail.parameters["track_usage"];
      }

      if (
        this.pendingAutoloadRequestConfig.suppressUrlChange &&
        !this.pendingAutoloadRequestConfig.trackUsage
      ) {
        htmxEvent.detail.parameters["push_url"] = "false";
      } else {
        delete htmxEvent.detail.parameters["push_url"];
      }
      this.pendingAutoloadRequestConfig = null;
    });

    document.body.addEventListener("htmx:beforeRequest", (evt: Event) => {
      const htmxEvent = evt as HtmxBeforeRequestEvent;
      if (this.isResultsTarget(htmxEvent.detail.target)) {
        if (!this.isAutoloadRequest(htmxEvent.detail.elt)) {
          this.cancelInitialResultsAutoload();
        }
        this.showLoadingIndicator();
      }
    });

    // Handle HTMX after request completes - fade out spinner smoothly
    document.body.addEventListener("htmx:afterRequest", (evt: Event) => {
      const htmxEvent = evt as HtmxAfterRequestEvent;
      if (this.isResultsTarget(htmxEvent.detail.target)) {
        this.hideLoadingIndicator();
        if (this.isAutoloadRequest(htmxEvent.detail.elt)) {
          this.completeInitialResultsAutoload();
        }
      }
    });

    // Handle HTMX after swap for results visibility
    document.body.addEventListener("htmx:afterSwap", (evt: Event) => {
      const htmxEvent = evt as HtmxAfterSwapEvent;
      if (this.isResultsTarget(htmxEvent.detail.target)) {
        this.handleResultsSwap();
      }
    });

    document.body.addEventListener("htmx:afterSettle", (evt: Event) => {
      const htmxEvent = evt as HtmxAfterSettleEvent;
      if (this.isResultsTarget(htmxEvent.detail.target)) {
        this.handleResultsSettle();
      }
    });

    // Handle rate limit responses (429)
    document.body.addEventListener("htmx:responseError", (evt: Event) => {
      const htmxEvent = evt as HtmxResponseErrorEvent;

      if (htmxEvent.detail.xhr.status === 429) {
        if (this.isResultsTarget(htmxEvent.detail.target)) {
          // Display the paywall/error content returned by the server
          htmxEvent.detail.target.innerHTML = htmxEvent.detail.xhr.response;

          this.ensureResultsSectionVisible();
          this.hideLoadingIndicator();
          if (this.isAutoloadRequest(htmxEvent.detail.elt)) {
            this.completeInitialResultsAutoload();
          }
        }
      }
    });

    document.body.addEventListener("htmx:sendAbort", () => {
      this.clearAutoloadRequestState();
      this.hideLoadingIndicator();
    });

    document.body.addEventListener("htmx:timeout", () => {
      this.clearAutoloadRequestState();
      this.hideLoadingIndicator();
    });

    document.body.addEventListener("htmx:beforeHistorySave", () => {
      this.syncHistoryState();
    });

    document.body.addEventListener("htmx:beforeHistoryUpdate", (evt: Event) => {
      if (!this.suppressNextHistoryUpdate) {
        return;
      }

      const htmxEvent = evt as CustomEvent<{
        history?: { type?: string; path?: string };
      }>;
      if (!htmxEvent.detail.history) {
        return;
      }

      htmxEvent.detail.history.type = "replace";
      htmxEvent.detail.history.path =
        window.location.pathname +
        window.location.search +
        window.location.hash;
      this.suppressNextHistoryUpdate = false;
    });

    document.body.addEventListener("htmx:historyRestore", () => {
      this.hideLoadingIndicator();
      this.handleResultsSwap();
      this.handleResultsSettle();
    });

    window.addEventListener("pageshow", () => {
      this.hideLoadingIndicator();
    });
  }

  /**
   * Attach click listener to the share button
   * Called after HTMX swaps in the results
   */
  private attachShareButtonListener(): void {
    const shareButton = document.getElementById("share-button");
    if (shareButton) {
      // Remove any existing listeners to avoid duplicates
      const newButton = shareButton.cloneNode(true) as HTMLElement;
      shareButton.parentNode?.replaceChild(newButton, shareButton);

      // Add the click listener
      newButton.addEventListener("click", () => {
        this.copyShareableLink();
      });
    }
  }

  /**
   * Setup description toggle button functionality
   * Toggles the visibility of the description content
   */
  private setupDescriptionToggle(): void {
    const elements = this.getDescriptionToggleElements();
    if (!elements) return;

    if (!this.hasDescriptionText(elements.descriptionContent)) {
      this.hideDescriptionBlock(elements);
      return;
    }

    const learnMoreText = this.getDescriptionLearnMoreText(
      elements.toggleButton,
    );
    const isExpanded = this.isDescriptionExpanded(elements.toggleButton);
    this.applyDescriptionState(elements, isExpanded, learnMoreText);
    this.bindDescriptionToggle(elements, learnMoreText);
  }

  private getDescriptionToggleElements(): DescriptionToggleElements | null {
    const toggleButton = document.getElementById(
      "description-toggle",
    ) as HTMLButtonElement | null;
    const descriptionContent = document.getElementById(
      "description-content",
    ) as HTMLElement | null;
    const container = document.getElementById("description-container");

    if (!toggleButton || !descriptionContent || !container) return null;

    return { toggleButton, descriptionContent, container };
  }

  private hasDescriptionText(descriptionContent: HTMLElement): boolean {
    const text = descriptionContent.textContent ?? "";
    return Boolean(text.trim());
  }

  private hideDescriptionBlock({
    toggleButton,
    container,
  }: DescriptionToggleElements): void {
    toggleButton.style.display = "none";
    container.style.display = "none";
  }

  private getDescriptionLearnMoreText(toggleButton: HTMLButtonElement): string {
    const classifierType =
      toggleButton.getAttribute("data-classifier-type") || "";
    return classifierType ? `Learn more about ${classifierType}` : "Learn more";
  }

  private isDescriptionExpanded(toggleButton: HTMLButtonElement): boolean {
    return toggleButton.getAttribute("aria-expanded") === "true";
  }

  private bindDescriptionToggle(
    elements: DescriptionToggleElements,
    learnMoreText: string,
  ): void {
    elements.toggleButton.addEventListener("click", () => {
      const newExpandedState = !this.isDescriptionExpanded(
        elements.toggleButton,
      );
      elements.toggleButton.setAttribute(
        "aria-expanded",
        String(newExpandedState),
      );
      this.applyDescriptionState(elements, newExpandedState, learnMoreText);
    });
  }

  private applyDescriptionState(
    { toggleButton, descriptionContent }: DescriptionToggleElements,
    isExpanded: boolean,
    learnMoreText: string,
  ): void {
    descriptionContent.style.display = isExpanded ? "block" : "none";
    descriptionContent.setAttribute("aria-hidden", String(!isExpanded));
    toggleButton.textContent = isExpanded ? "Show less" : learnMoreText;
    this.setClassifierLogosVisible(!isExpanded);
  }

  private setClassifierLogosVisible(visible: boolean): void {
    const logoElements = document.querySelectorAll(
      '[data-classifier-logo="true"]',
    ) as NodeListOf<HTMLElement>;
    logoElements.forEach((logo) => {
      logo.style.display = visible ? "" : "none";
    });
  }

  /**
   * Copy the current page URL to clipboard
   * Exposed globally for inline onclick handlers
   */
  public copyShareableLink(): void {
    ShareLink.copyShareableLink();
  }
}

// Initialize classifier page functionality when DOM is ready
export function initClassifierPage(): void {
  new ClassifierPage();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initClassifierPage);
} else {
  initClassifierPage();
}
