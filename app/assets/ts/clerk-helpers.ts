import "./types/globals";

/**
 * Shared Clerk authentication utilities to avoid code duplication
 * across paywall and common modules.
 */

export class ClerkHelpers {
  /**
   * Open Clerk sign-in modal with fallback redirect
   */
  static openSignIn(fallbackButtonId?: string): void {
    if (window.Clerk?.openSignIn) {
      window.Clerk.openSignIn({ redirectUrl: window.location.href });
    } else {
      // Fallback: redirect to accounts page
      const redirectUrl = encodeURIComponent(window.location.href);
      window.location.href = `https://accounts.classifast.com/sign-in?redirect_url=${redirectUrl}`;
    }
  }

  /**
   * Open Clerk sign-up modal with fallback redirect
   */
  static openSignUp(): void {
    if (window.Clerk?.openSignUp) {
      window.Clerk.openSignUp({ redirectUrl: window.location.href });
    } else {
      // Fallback: redirect to accounts page
      const redirectUrl = encodeURIComponent(window.location.href);
      window.location.href = `https://accounts.classifast.com/sign-up?redirect_url=${redirectUrl}`;
    }
  }

  /**
   * Create error message element for auth failures
   */
  static createAuthErrorMessage(): HTMLDivElement {
    const errorDiv = document.createElement("div");
    errorDiv.className =
      "bg-amber-50 border border-amber-200 rounded-lg p-3 mt-3 text-center";
    errorDiv.innerHTML = `
      <p class="text-sm text-amber-800">
        <svg class="w-4 h-4 inline mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-label="Warning">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        Authentication service is temporarily unavailable. Redirecting...
      </p>
    `;
    return errorDiv;
  }

  /**
   * Show auth error and redirect with delay
   */
  static showAuthErrorAndRedirect(
    containerId: string,
    action: "sign-in" | "sign-up",
    fallbackUrl?: string,
    delayMs: number = 2000,
  ): void {
    const container = document.getElementById(containerId);
    if (container) {
      container.appendChild(this.createAuthErrorMessage());
    }

    const targetUrl =
      fallbackUrl || `https://accounts.classifast.com/${action}`;

    setTimeout(() => {
      try {
        const url = new URL(targetUrl, window.location.origin);
        url.searchParams.set("redirect_url", window.location.href);
        window.location.href = url.toString();
      } catch (err: unknown) {
        console.error("Error parsing URL, using fallback:", err);
        const separator = targetUrl.includes("?") ? "&" : "?";
        window.location.href = `${targetUrl}${separator}redirect_url=${encodeURIComponent(window.location.href)}`;
      }
    }, delayMs);
  }

  /**
   * Safely submit a form by selector
   * Returns true if form was found and submitted, false otherwise
   */
  static submitForm(selector: string = "form[hx-get]"): boolean {
    const form = document.querySelector(selector) as HTMLFormElement | null;
    if (form && typeof form.requestSubmit === "function") {
      form.requestSubmit();
      return true;
    }
    return false;
  }
}
