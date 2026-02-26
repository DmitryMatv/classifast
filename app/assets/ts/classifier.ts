import { ShareLink } from "./common";

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
    this.setupDescriptionToggle();
    this.setCursorToEnd();
  }

  private setCursorToEnd(): void {
    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement | null;
    if (textarea && textarea.value) {
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
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
    // Handle HTMX after request completes - fade out spinner smoothly
    document.body.addEventListener("htmx:afterRequest", (evt: Event) => {
      const htmxEvent = evt as HtmxAfterRequestEvent;
      const indicator = document.getElementById("loading-indicator");
      if (indicator && htmxEvent.detail.target.id === "results-container") {
        indicator.classList.remove("htmx-request");
      }
    });

    // Handle HTMX after swap for results visibility
    document.body.addEventListener("htmx:afterSwap", (evt: Event) => {
      const htmxEvent = evt as HtmxAfterSwapEvent;
      if (htmxEvent.detail.target.id === "results-container") {
        const resultsSection = document.getElementById("results-section");
        if (resultsSection) {
          resultsSection.classList.remove("hidden");
        }
        // Attach share button listener after results are swapped in
        this.attachShareButtonListener();
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
  new ClassifierPage();
});

// Also run immediately if DOM is already loaded
if (
  document.readyState === "complete" ||
  document.readyState === "interactive"
) {
  new ClassifierPage();
}

// Run immediately if DOM is already ready, otherwise the listener above will handle it
if (
  document.readyState === "complete" ||
  document.readyState === "interactive"
) {
  showInitialLoadingIndicator();
}

// Also expose the loading indicator function for template use
window.showInitialLoadingIndicator = showInitialLoadingIndicator;
