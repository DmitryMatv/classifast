import "./types/globals";

/**
 * Shared Clerk authentication utilities to avoid code duplication
 * across paywall and common modules.
 */

export class ClerkHelpers {
  /**
   * Open Clerk sign-in modal with fallback redirect
   */
  static openSignIn(): void {
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
