import { ShareLink, ResultCopier } from "./common";

/**
 * Classifier page specific functionality
 * Handles form auto-submission, HTMX event handling, and UI interactions
 */

class ClassifierPage {
  constructor() {
    this.init();
  }

  private init(): void {
    this.setupTopKAutosubmit();
    this.setupHTMXListeners();
    // Initialize ResultCopier for copy functionality
    new ResultCopier();
  }

  /**
   * Setup automatic form submission when Top K selector changes
   * Only submits if there's text in the description area
   */
  private setupTopKAutosubmit(): void {
    const topKSelector = document.getElementById(
      "show_top_k_categories",
    ) as HTMLSelectElement | null;
    const productDescriptionArea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement | null;

    if (topKSelector && productDescriptionArea) {
      topKSelector.addEventListener("change", () => {
        if (productDescriptionArea.value.trim()) {
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

  /**
   * Setup HTMX event listeners for response handling
   */
  private setupHTMXListeners(): void {
    // Handle HTMX after swap for results visibility
    document.body.addEventListener("htmx:afterSwap", (evt: Event) => {
      const htmxEvent = evt as HtmxAfterSwapEvent;
      if (htmxEvent.detail.target.id === "results-container") {
        const resultsSection = document.getElementById("results-section");
        if (resultsSection) {
          resultsSection.classList.remove("hidden");
        }
      }
    });

    // Handle rate limit responses (429)
    document.body.addEventListener("htmx:responseError", (evt: Event) => {
      const htmxEvent = evt as HtmxResponseErrorEvent;

      if (htmxEvent.detail.xhr.status === 429) {
        if (htmxEvent.detail.target.id === "results-container") {
          // Display the paywall/error content returned by the server
          htmxEvent.detail.target.innerHTML = htmxEvent.detail.xhr.response;

          const resultsSection = document.getElementById("results-section");
          if (resultsSection) {
            resultsSection.classList.remove("hidden");
          }

          const loadingIndicator = document.getElementById("loading-indicator");
          if (loadingIndicator) {
            loadingIndicator.classList.remove("htmx-request");
          }
        }
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

/**
 * Show loading indicator immediately while waiting for auth
 * This runs immediately when the script loads to avoid delay
 */
function showInitialLoadingIndicator(): void {
  const indicator = document.getElementById("loading-indicator");
  if (indicator) {
    indicator.classList.add("htmx-request");
  }
}

// Initialize classifier page functionality when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  // Initialize the classifier page
  new ClassifierPage();
});

// Run immediately if DOM is already ready, otherwise the listener above will handle it
if (
  document.readyState === "complete" ||
  document.readyState === "interactive"
) {
  showInitialLoadingIndicator();
}

// Also expose the loading indicator function for template use
window.showInitialLoadingIndicator = showInitialLoadingIndicator;
