import "./types/globals";
import { ClerkHelpers } from "./clerk-helpers";

/**
 * Paywall-specific functionality for handling authentication,
 * checkout flows, and automatic form resubmission after auth changes.
 */

class PaywallManager {
  private wasSignedIn: boolean = false;

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

  /**
   * Setup retry button click handler
   */
  private setupRetryButton(): void {
    const retryButton = document.getElementById("retry-button");
    if (retryButton) {
      retryButton.addEventListener("click", () =>
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

    if (window.Clerk?.addListener) {
      const unsubscribe = window.Clerk.addListener((resources) => {
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

      // Store unsubscribe for cleanup if needed (e.g., in SPA navigation)
      // For now, this is fine as the paywall is replaced via HTMX
    }
  }

  /**
   * Setup sign in/up button handlers for anonymous users
   */
  private setupAuthButtons(): void {
    const signinButton = document.getElementById("signin-button");
    const signupButton = document.getElementById("signup-button");

    if (signinButton) {
      signinButton.addEventListener("click", (e) => {
        e.preventDefault();
        if (window.Clerk?.openSignIn) {
          window.Clerk.openSignIn({ redirectUrl: window.location.href });
        } else {
          const fallbackUrl = signinButton.dataset.fallbackUrl;
          ClerkHelpers.showAuthErrorAndRedirect(
            "paywall-buttons",
            "sign-in",
            fallbackUrl,
          );
        }
      });
    }

    if (signupButton) {
      signupButton.addEventListener("click", (e) => {
        e.preventDefault();
        if (window.Clerk?.openSignUp) {
          window.Clerk.openSignUp({ redirectUrl: window.location.href });
        } else {
          const fallbackUrl = signupButton.dataset.fallbackUrl;
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

    upgradeButton.addEventListener("click", async (e) => {
      e.preventDefault();
      await this.handleUpgrade(upgradeButton as HTMLButtonElement);
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
          product_id: "e157e32f-e91c-4d51-af66-0c2eb3b23d71",
          return_url: window.location.href,
        }),
      });

      clearTimeout(timeoutId);

      if (!response.ok) throw new Error("Checkout creation failed");

      const data = (await response.json()) as { url?: string };
      if (data.url) {
        window.location.href = data.url;
      } else {
        throw new Error("No checkout URL returned");
      }
    } catch (err) {
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
function initPaywall(): void {
  // Always initialize the paywall manager (handles retry + auth transitions)
  new PaywallManager();

  // Initialize checkout manager only if upgrade button exists (authenticated users)
  const upgradeButton = document.getElementById("upgrade-button");
  if (upgradeButton) {
    new CheckoutManager();
  }
}

// Prevent duplicate initialization across DOMContentLoaded and immediate execution
if (!window.__paywallInitialized) {
  window.__paywallInitialized = true;

  if (document.readyState === "loading") {
    // DOM not ready yet, wait for it
    document.addEventListener("DOMContentLoaded", initPaywall);
  } else {
    // DOM already ready, initialize immediately with small delay for HTMX swaps
    setTimeout(initPaywall, 0);
  }
}
