import { ShareLink } from "./common";

const BASE_SCORE_BAR_DELAY_MS = 0;
const SCORE_BAR_STAGGER_MS = 100;
const MAX_SCORE_BAR_STAGGER_MS = 600;

/**
 * Classifier page specific functionality
 * Handles form auto-submission, HTMX event handling, and UI interactions
 */

class ClassifierPage {
  private allowFragmentSubmit = false;

  constructor() {
    this.init();
  }

  private init(): void {
    document.documentElement.classList.add("js-score-animations");
    this.setupSearchNavigation();
    this.setupTopKAutosubmit();
    this.setupHTMXListeners();
    this.setupDescriptionToggle();
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

  private getInitialResultsLoader(): HTMLElement | null {
    return document.querySelector("[data-initial-results-loader='true']");
  }

  private getClassifierForm(): HTMLFormElement | null {
    const form = document.querySelector("form[hx-get]");
    return form instanceof HTMLFormElement ? form : null;
  }

  private getProductDescriptionArea(): HTMLTextAreaElement | null {
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement | null;
    return textarea;
  }

  private getVersionSelector(): HTMLSelectElement | null {
    return document.getElementById(
      "version_selector",
    ) as HTMLSelectElement | null;
  }

  private cleanupInitialResultsLoader(): void {
    this.getInitialResultsLoader()?.remove();
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
    const productDescriptionArea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement | null;

    if (!productDescriptionArea) {
      return;
    }

    productDescriptionArea.defaultValue = productDescriptionArea.value;
    productDescriptionArea.textContent = productDescriptionArea.value;
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
    this.cleanupInitialResultsLoader();
  }

  private normalizeFragmentQueryInput(text: string): string {
    return text.normalize("NFC").replace(/\s+/g, " ").trim();
  }

  private normalizeProductDescriptionInPlace(): string {
    const productDescriptionArea = this.getProductDescriptionArea();
    const normalizedDescription = this.normalizeFragmentQueryInput(
      productDescriptionArea?.value ?? "",
    );

    if (productDescriptionArea) {
      productDescriptionArea.value = normalizedDescription;
    }

    return normalizedDescription;
  }

  private buildSearchNavigationUrl(form: HTMLFormElement): string {
    const formData = new FormData(form);
    const params = new URLSearchParams();

    for (const [key, value] of formData.entries()) {
      if (key === "track_usage") {
        continue;
      }

      const stringValue = String(value);
      if (key === "product_description" && !stringValue) {
        continue;
      }
      params.append(key, stringValue);
    }

    const queryString = params.toString();
    return queryString ? `${form.action}?${queryString}` : form.action;
  }

  private navigate(url: string): void {
    if (window.__classifierNavigate) {
      window.__classifierNavigate(url);
      return;
    }

    window.location.assign(url);
  }

  private navigateToSearchUrl(): void {
    const form = this.getClassifierForm();
    if (!form) {
      return;
    }

    this.normalizeProductDescriptionInPlace();
    this.navigate(this.buildSearchNavigationUrl(form));
  }

  private setupSearchNavigation(): void {
    const form = this.getClassifierForm();
    const versionSelector = this.getVersionSelector();

    if (!form) {
      return;
    }

    form.addEventListener(
      "submit",
      (event) => {
        if (this.allowFragmentSubmit) {
          this.allowFragmentSubmit = false;
          return;
        }

        event.preventDefault();
        this.navigateToSearchUrl();
      },
      true,
    );

    versionSelector?.addEventListener("change", () => {
      const normalizedDescription = this.normalizeProductDescriptionInPlace();
      if (normalizedDescription) {
        this.navigateToSearchUrl();
      }
    });
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
    this.cleanupInitialResultsLoader();
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
        const normalizedDescription = this.normalizeProductDescriptionInPlace();
        if (normalizedDescription) {
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
        this.allowFragmentSubmit = true;
        window.htmx.trigger(form, "submit");
      }
    }
  }

  /**
   * Setup HTMX event listeners for response handling
   */
  private setupHTMXListeners(): void {
    document.body.addEventListener("htmx:beforeRequest", (evt: Event) => {
      const htmxEvent = evt as HtmxBeforeRequestEvent;
      if (this.isResultsTarget(htmxEvent.detail.target)) {
        this.showLoadingIndicator();
      }
    });

    // Handle HTMX after request completes - fade out spinner smoothly
    document.body.addEventListener("htmx:afterRequest", (evt: Event) => {
      const htmxEvent = evt as HtmxAfterRequestEvent;
      if (this.isResultsTarget(htmxEvent.detail.target)) {
        this.hideLoadingIndicator();
      }

      if (
        htmxEvent.detail.elt instanceof HTMLElement &&
        htmxEvent.detail.elt.hasAttribute("data-initial-results-loader")
      ) {
        this.cleanupInitialResultsLoader();
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
          this.cleanupInitialResultsLoader();
        }
      }
    });

    document.body.addEventListener("htmx:sendAbort", () => {
      this.hideLoadingIndicator();
    });

    document.body.addEventListener("htmx:timeout", () => {
      this.hideLoadingIndicator();
    });

    document.body.addEventListener("htmx:beforeHistorySave", () => {
      this.syncHistoryState();
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
    const toggleButton = document.getElementById(
      "description-toggle",
    ) as HTMLButtonElement | null;
    const descriptionContent = document.getElementById(
      "description-content",
    ) as HTMLElement | null;
    const container = document.getElementById("description-container");

    if (!toggleButton || !descriptionContent || !container) return;

    // Hide entire block if description empty
    const text = descriptionContent.textContent ?? "";
    if (!text.trim()) {
      toggleButton.style.display = "none";
      container.style.display = "none";
      return;
    }

    const classifierType =
      toggleButton.getAttribute("data-classifier-type") || "";
    const learnMoreText = classifierType
      ? `Learn more about ${classifierType}`
      : "Learn more";
    const isExpanded = toggleButton.getAttribute("aria-expanded") === "true";
    descriptionContent.style.display = isExpanded ? "block" : "none";
    descriptionContent.setAttribute("aria-hidden", String(!isExpanded));
    toggleButton.textContent = isExpanded ? "Show less" : learnMoreText;

    const initialLogoElements = document.querySelectorAll(
      '[data-classifier-logo="true"]',
    ) as NodeListOf<HTMLElement>;
    initialLogoElements.forEach((logo) => {
      logo.style.display = isExpanded ? "none" : "";
    });

    toggleButton.addEventListener("click", () => {
      const currentlyExpanded =
        toggleButton.getAttribute("aria-expanded") === "true";
      const newExpandedState = !currentlyExpanded;

      toggleButton.setAttribute("aria-expanded", String(newExpandedState));

      const currentLogoElements = document.querySelectorAll(
        '[data-classifier-logo="true"]',
      ) as NodeListOf<HTMLElement>;

      if (newExpandedState) {
        descriptionContent.style.display = "block";
        descriptionContent.setAttribute("aria-hidden", "false");
        toggleButton.textContent = "Show less";
        currentLogoElements.forEach((logo) => {
          logo.style.display = "none";
        });
      } else {
        descriptionContent.style.display = "none";
        descriptionContent.setAttribute("aria-hidden", "true");
        toggleButton.textContent = learnMoreText;
        currentLogoElements.forEach((logo) => {
          logo.style.display = "";
        });
      }
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
