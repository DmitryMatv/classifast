import "./types/globals";
import { ClerkHelpers } from "./clerk-helpers";

/**
 * Paywall-specific functionality for handling authentication,
 * checkout flows, and automatic form resubmission after auth changes.
 *
 * CRITICAL: This entire script is wrapped in a parse guard to prevent
 * "Identifier 'X' has already been declared" errors when the script
 * is re-parsed (e.g., browser back/forward navigation, HTMX history restoration).
 * Class declarations execute at parse time, so we must prevent re-parsing.
 */

// Guard: Prevent duplicate script parsing (classes can only be declared once)
export function initPaywall(): void {
  window.__initPaywall?.();
}

function redirectToPaywallUrl(url: string): void {
  if (window.__paywallNavigate) {
    window.__paywallNavigate(url);
    return;
  }

  window.location.assign(url);
}

if (!window.__paywallScriptParsed) {
  window.__paywallScriptParsed = true;

  const PAYWALL_PRODUCT_ID: string = "e157e32f-e91c-4d51-af66-0c2eb3b23d71";

  class PaywallManager {
    private wasSignedIn: boolean = false;
    private authReadyRetryHandled: boolean = false;

    constructor() {
      this.init();
    }

    private init() {
      this.setupRetryButton();
      this.setupClerkListener();
      this.setupAuthButtons();
    }

    /**
     * Helper to safely submit the classification form
     */
    private submitClassificationForm(): void {
      ClerkHelpers.submitForm("form[hx-get]");
    }

    private submitClassificationFormOnce(): void {
      if (this.authReadyRetryHandled) {
        return;
      }

      this.authReadyRetryHandled = true;
      this.submitClassificationForm();
    }

    private registerClerkAuthTransitionListener(): boolean {
      if (!window.Clerk?.addListener) {
        return false;
      }

      window.Clerk.addListener((resources) => {
        // Only auto-retry when user transitions from signed-out to signed-in
        if (resources.user && !this.wasSignedIn) {
          this.wasSignedIn = true; // Prevent repeated submissions on token refresh
          this.submitClassificationForm();
        }
        // If user signs out, reset the flag
        if (!resources.user) {
          this.wasSignedIn = false;
        }
      });

      return true;
    }

    /**
     * Setup retry button click handler
     */
    private setupRetryButton(): void {
      const retryButton = document.getElementById("retry-button");
      if (retryButton) {
        // Remove existing listener to prevent duplicates on re-initialization
        const newRetryButton = retryButton.cloneNode(true) as HTMLElement;
        retryButton.parentNode?.replaceChild(newRetryButton, retryButton);
        newRetryButton.addEventListener("click", () =>
          this.submitClassificationForm(),
        );
      }
    }

    /**
     * Setup Clerk auth listener for auto-retry after login/signup
     * This handles the transition from anonymous to authenticated user
     */
    private setupClerkListener(): void {
      // Guard: Only register once globally to prevent accumulating listeners
      if (window.__paywallClerkListenerRegistered) {
        return;
      }
      window.__paywallClerkListenerRegistered = true;

      // Track the initial user state when paywall loads
      this.wasSignedIn = !!(window.Clerk && window.Clerk.user);

      if (!this.registerClerkAuthTransitionListener()) {
        document.body.addEventListener(
          "htmx:authReady",
          () => {
            this.wasSignedIn = !!window.Clerk?.user;
            this.registerClerkAuthTransitionListener();
            if (window.Clerk?.user) {
              this.submitClassificationFormOnce();
            }
          },
          { once: true },
        );
        return;
      }

      // Store unsubscribe for cleanup if needed (e.g., in SPA navigation)
      // For now, this is fine as the paywall is replaced via HTMX
    }

    /**
     * Setup sign in/up button handlers for anonymous users
     */
    private setupAuthButtons(): void {
      const signinButton = document.getElementById("signin-button");
      const signupButton = document.getElementById("signup-button");

      if (signinButton) {
        // Remove existing listener to prevent duplicates
        const newSigninButton = signinButton.cloneNode(true) as HTMLElement;
        signinButton.parentNode?.replaceChild(newSigninButton, signinButton);
        newSigninButton.addEventListener("click", (e) => {
          e.preventDefault();
          if (window.Clerk?.openSignIn) {
            window.Clerk.openSignIn({ redirectUrl: window.location.href });
          } else {
            const fallbackUrl = newSigninButton.dataset["fallbackUrl"];
            ClerkHelpers.showAuthErrorAndRedirect(
              "paywall-buttons",
              "sign-in",
              fallbackUrl,
            );
          }
        });
      }

      if (signupButton) {
        // Remove existing listener to prevent duplicates
        const newSignupButton = signupButton.cloneNode(true) as HTMLElement;
        signupButton.parentNode?.replaceChild(newSignupButton, signupButton);
        newSignupButton.addEventListener("click", (e) => {
          e.preventDefault();
          if (window.Clerk?.openSignUp) {
            window.Clerk.openSignUp({ redirectUrl: window.location.href });
          } else {
            const fallbackUrl = newSignupButton.dataset["fallbackUrl"];
            ClerkHelpers.showAuthErrorAndRedirect(
              "paywall-buttons",
              "sign-up",
              fallbackUrl,
            );
          }
        });
      }
    }
  }

  /**
   * Checkout manager for Pro upgrade flow
   */
  class CheckoutManager {
    constructor() {
      this.init();
    }

    private init() {
      this.setupUpgradeButton();
    }

    private setupUpgradeButton(): void {
      const upgradeButton = document.getElementById("upgrade-button");
      if (!upgradeButton) return;

      // Remove existing listener to prevent duplicates on re-initialization
      const newUpgradeButton = upgradeButton.cloneNode(
        true,
      ) as HTMLButtonElement;
      upgradeButton.parentNode?.replaceChild(newUpgradeButton, upgradeButton);

      newUpgradeButton.addEventListener("click", async (e) => {
        e.preventDefault();
        await this.handleUpgrade(newUpgradeButton);
      });
    }

    private async handleUpgrade(button: HTMLButtonElement): Promise<void> {
      if (!window.Clerk?.session) {
        console.error("Clerk not available");
        this.showErrorState(button);
        return;
      }

      try {
        this.showLoadingState(button);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        try {
          const token = await window.Clerk.session.getToken();
          if (!token) {
            console.error("Failed to get auth token");
            this.showErrorState(button);
            return;
          }
          const response = await fetch("/api/create-checkout", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            signal: controller.signal,
            body: JSON.stringify({
              product_id: PAYWALL_PRODUCT_ID,
              return_url: window.location.href,
            }),
          });

          if (!response.ok) throw new Error("Checkout creation failed");

          const data = (await response.json()) as { url?: string };
          if (data.url) {
            redirectToPaywallUrl(data.url);
          } else {
            throw new Error("No checkout URL returned");
          }
        } finally {
          clearTimeout(timeoutId);
        }
      } catch (err: unknown) {
        this.handleError(err, button);
      }
    }

    private showLoadingState(button: HTMLButtonElement): void {
      button.disabled = true;
      button.innerHTML = `
      <svg class="w-4 h-4 mr-2 inline animate-spin" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>Preparing...
    `;
    }

    private showErrorState(button: HTMLButtonElement): void {
      button.innerHTML = `
      <svg class="w-4 h-4 mr-2 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>Error - Try again
    `;
      button.disabled = false;
    }

    private handleError(err: unknown, button: HTMLButtonElement): void {
      if (err instanceof Error && err.name === "AbortError") {
        console.error("Checkout request timed out");
      } else {
        console.error("Upgrade failed:", err);
      }

      this.showErrorState(button);

      setTimeout(() => {
        button.disabled = false;
        button.innerHTML = `
        <svg class="w-4 h-4 mr-2 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>Upgrade to Pro
      `;
      }, 3000);
    }
  }

  // Initialize paywall functionality when DOM is ready
  function initPaywallImpl(): void {
    // Guard: Prevent duplicate initialization
    if (window.__paywallInitialized) {
      return;
    }
    window.__paywallInitialized = true;

    // Check if paywall content exists
    const paywallWarning = document.getElementById("paywall-warning");
    const paywallButtons = document.getElementById("paywall-buttons");

    // Only initialize if paywall elements exist in the DOM
    if (!paywallWarning || !paywallButtons) {
      return;
    }

    // Always initialize the paywall manager (handles retry + auth transitions)
    new PaywallManager();

    // Initialize checkout manager only if upgrade button exists (authenticated users)
    const upgradeButton = document.getElementById("upgrade-button");
    if (upgradeButton) {
      new CheckoutManager();
    }
  }

  window.__initPaywall = initPaywallImpl;

  // Initialize on DOMContentLoaded (first page load)
  document.addEventListener("DOMContentLoaded", () => {
    initPaywallImpl();
  });

  // Initialize on HTMX afterSwap (when paywall is swapped into results-container)
  document.body.addEventListener("htmx:afterSwap", (evt: Event) => {
    const htmxEvent = evt as HtmxAfterSwapEvent;
    // Only initialize if the results-container was updated
    if (htmxEvent.detail.target.id === "results-container") {
      // Reset initialization flag to allow re-initialization with new DOM elements
      window.__paywallInitialized = false;
      // Small delay to ensure DOM is updated
      setTimeout(initPaywallImpl, 0);
    }
  });

  // Also try to initialize immediately in case DOM is already ready
  // and this script loaded after HTMX swapped content
  if (document.readyState !== "loading") {
    setTimeout(initPaywallImpl, 0);
  }
} // End of __paywallScriptParsed guard
