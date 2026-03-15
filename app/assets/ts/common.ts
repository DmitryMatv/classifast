import "./types/globals";
import { ClerkHelpers } from "./clerk-helpers";

// Shared TypeScript functionality for Classifast application

const SIGN_IN_CLASS =
  "bg-sky-50 text-sky-700 hover:bg-sky-100 active:bg-sky-100 active:scale-95 rounded transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded";
const SIGN_UP_CLASS =
  "bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white rounded transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded";
const DESKTOP_AUTH_BUTTON_SIZE_CLASS = "px-5 py-1.5";
const MOBILE_AUTH_BUTTON_SIZE_CLASS = "px-4 py-1";

// Global error handlers
window.addEventListener("error", (event) => {
  console.error("Global error:", event.error);
});

window.addEventListener("unhandledrejection", (event) => {
  console.error("Unhandled promise rejection:", event.reason);
});

// Mobile menu functionality
export class MobileMenu {
  private button: HTMLElement | null = null;
  private menu: HTMLElement | null = null;
  private hamburger: HTMLElement | null = null;

  constructor() {
    this.init();
  }

  private init() {
    this.button = document.getElementById("mobile-menu-button");
    this.menu = document.getElementById("mobile-menu");
    this.hamburger = document.querySelector(".hamburger");

    if (!this.button || !this.menu || !this.hamburger) return;

    this.button.addEventListener("click", () => this.toggle());

    // Close on link click
    const links = this.menu.querySelectorAll("a");
    links.forEach((link) => {
      link.addEventListener("click", () => this.close());
    });

    // Close on outside click
    document.addEventListener("click", (e) => {
      if (
        !this.menu?.contains(e.target as Node) &&
        !this.button?.contains(e.target as Node)
      ) {
        this.close();
      }
    });

    // Close on ESC key
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.menu?.classList.contains("active")) {
        this.close();
        this.button?.focus();
      }
    });
  }

  private toggle() {
    const isActive = this.menu?.classList.toggle("active") ?? false;
    this.hamburger?.classList.toggle("active");
    this.button?.setAttribute("aria-expanded", String(isActive));
  }

  private close() {
    this.menu?.classList.remove("active");
    this.hamburger?.classList.remove("active");
    this.button?.setAttribute("aria-expanded", "false");
  }
}

// Copy URL functionality
export class ShareLink {
  static async copyShareableLink() {
    const url = window.location.href;
    const button = document.getElementById("share-button");

    try {
      await navigator.clipboard.writeText(url);
      this.showFeedback(button);
    } catch (err: unknown) {
      console.error("Could not copy URL: ", err);
      this.fallbackCopy(url, button);
    }
  }

  private static showFeedback(button: HTMLElement | null) {
    if (!button) return;

    const originalText = button.innerHTML;
    button.innerHTML = "Copied!";
    button.classList.add("bg-green-600", "hover:bg-green-700");

    setTimeout(() => {
      button.innerHTML = originalText;
      button.classList.remove("bg-green-600", "hover:bg-green-700");
    }, 2000);
  }

  private static fallbackCopy(url: string, button: HTMLElement | null) {
    const textArea = document.createElement("textarea");
    textArea.value = url;
    document.body.appendChild(textArea);
    textArea.select();

    try {
      document.execCommand("copy");
      console.log("URL copied using fallback");
      this.showFeedback(button);
    } catch (fallbackErr: unknown) {
      console.error("Fallback copy failed: ", fallbackErr);
    }

    document.body.removeChild(textArea);
  }
}

// Textarea enhanced functionality
export class TextareaEnhancer {
  private textarea: HTMLTextAreaElement | null;

  constructor(textareaId: string) {
    this.textarea = document.getElementById(
      textareaId,
    ) as HTMLTextAreaElement | null;
    if (this.textarea) {
      this.init();
    }
  }

  private init() {
    this.textarea?.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.submitForm();
      }
    });
  }

  private submitForm() {
    const form = this.textarea?.closest("form");
    if (form) {
      const submitBtn = form.querySelector(
        'button[type="submit"]',
      ) as HTMLElement | null;
      if (submitBtn) {
        submitBtn.classList.add("active", "scale-95");
        setTimeout(() => {
          submitBtn.classList.remove("active", "scale-95");
        }, 150);
        submitBtn.click();
      } else {
        form.requestSubmit();
      }
    }
  }

  getValue() {
    return this.textarea?.value ?? "";
  }

  setValue(value: string) {
    if (this.textarea) {
      this.textarea.value = value;
    }
  }
}

// Cached auth token for synchronous HTMX header injection
let cachedAuthToken: string | null = null;

function isTokenExpired(token: string): boolean {
  try {
    const parts = token.split(".");
    const payloadB64 = parts[1];
    if (parts.length !== 3 || !payloadB64) return true;

    const base64 = payloadB64.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64 + "==".slice(0, (4 - (base64.length % 4)) % 4);

    const payload = JSON.parse(atob(padded)) as { exp?: number };
    if (!payload.exp) return true;
    const now = Math.floor(Date.now() / 1000);
    return payload.exp < now;
  } catch (err: unknown) {
    console.error("Error parsing JWT token:", err);
    return true;
  }
}

// Track if auth-ready event has been fired (fire only once on initial load)
let authReadyFired = false;

function signalAuthReady(): void {
  if (authReadyFired) return;

  authReadyFired = true;
  window.__clerkAuthReady = true;
  document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
}

// Simple Clerk Authentication using official SDK patterns
export class ClerkAuth {
  private static htmxAuthHeaderRegistered = false;
  private clerkStarted = false;
  private readonly clerkScriptSelector =
    'script[src*="@clerk/clerk-js"], script[src*="clerk.browser.js"]';

  constructor() {
    this.init();
  }

  private async init() {
    if (window.Clerk) {
      await this.startClerk();
      return;
    }

    await this.waitForClerk();
  }

  private async startClerk() {
    if (this.clerkStarted) return;
    this.clerkStarted = true;

    try {
      await window.Clerk?.load();

      // Register HTMX header injection (must be after Clerk loads)
      this.registerHtmxAuthHeader();

      // Check if returning from successful checkout - clean up URL params
      const urlParams = new URLSearchParams(window.location.search);
      const isCheckoutSuccess = urlParams.get("checkout") === "success";

      if (isCheckoutSuccess) {
        // Clean up checkout params from URL
        urlParams.delete("checkout");
        urlParams.delete("checkout_token");
        urlParams.delete("customer_session_token");
        const cleanUrl = urlParams.toString()
          ? `${window.location.pathname}?${urlParams.toString()}`
          : window.location.pathname;
        window.history.replaceState({}, "", cleanUrl);
        // Backend verifies tier via Redis-cached Clerk API - no frontend retries needed
      }

      // Initial token cache and UI render
      await this.refreshAuthToken();
      this.updateAuthUI();

      // Signal that auth is ready for auto-classification (fire only once)
      signalAuthReady();

      // Refresh token every 50s (Clerk tokens expire in ~60s)
      // getToken with expirationBufferSeconds handles caching automatically
      // and only makes network requests when the token is near expiration
      setInterval(() => this.refreshAuthToken(), 50000);

      // Also refresh on user interaction to ensure token is fresh before requests
      // This is a backup in case the interval misses or page was inactive
      // Guard to prevent duplicate listener registration on re-initialization
      if (!window.__clerkInteractionListenersRegistered) {
        window.__clerkInteractionListenersRegistered = true;
        let refreshPending = false;
        const refreshOnInteraction = () => {
          const tokenExpired = cachedAuthToken
            ? isTokenExpired(cachedAuthToken)
            : true;
          if (
            window.Clerk?.session &&
            (!cachedAuthToken || tokenExpired) &&
            !refreshPending
          ) {
            refreshPending = true;
            console.log("Refreshing token on user interaction...");
            this.refreshAuthToken().finally(() => {
              refreshPending = false;
            });
          }
        };
        document.addEventListener("click", refreshOnInteraction, {
          passive: true,
        });
        document.addEventListener("keydown", refreshOnInteraction, {
          passive: true,
        });
      }

      // Listen for auth state changes (guard to prevent duplicate listeners)
      if (window.Clerk?.addListener && !window.__clerkAuthListenerRegistered) {
        window.__clerkAuthListenerRegistered = true;
        window.Clerk.addListener(async () => {
          await this.refreshAuthToken();
          this.updateAuthUI();
          // Note: We intentionally don't dispatch htmx:authReady here
          // Auto-classification should only happen on initial page load
        });
      }
    } catch (err: unknown) {
      this.clerkStarted = false;
      console.error("Error initializing Clerk:", err);
      this.renderFallbackAuth();
    }
  }

  private async waitForClerk() {
    if (window.Clerk) {
      await this.startClerk();
      return;
    }

    const script = document.querySelector(
      this.clerkScriptSelector,
    ) as HTMLScriptElement | null;

    if (!script) {
      console.error("Clerk script tag not found");
      this.renderFallbackAuth();
      return;
    }

    let settled = false;
    let timeoutId: number | null = null;

    const cleanup = () => {
      script.removeEventListener("load", onScriptLoaded);
      script.removeEventListener("error", onScriptError);
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    const settleWithFallback = (message: string) => {
      if (settled) return;
      settled = true;
      cleanup();
      console.error(message);
      this.renderFallbackAuth();
    };

    const onScriptLoaded = async () => {
      if (settled) return;
      const clerk = await this.pollForClerk();
      if (clerk) {
        settled = true;
        cleanup();
        await this.startClerk();
      } else {
        settleWithFallback(
          "Clerk script loaded but window.Clerk was unavailable",
        );
      }
    };

    const onScriptError = () => {
      settleWithFallback("Clerk script failed to load");
    };

    script.addEventListener("load", onScriptLoaded, { once: true });
    script.addEventListener("error", onScriptError, { once: true });

    const existingClerk = await this.pollForClerk(1000);
    if (existingClerk) {
      settled = true;
      cleanup();
      await this.startClerk();
      return;
    }

    timeoutId = window.setTimeout(async () => {
      if (settled) return;
      if (window.Clerk) {
        settled = true;
        cleanup();
        await this.startClerk();
        return;
      }

      settleWithFallback("Timed out waiting for Clerk script readiness");
    }, 4000);
  }

  private async pollForClerk(timeoutMs = 4000) {
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      if (window.Clerk) return window.Clerk;
      await new Promise((resolve) => window.setTimeout(resolve, 50));
    }

    return window.Clerk;
  }

  private async refreshAuthToken() {
    await ClerkAuth.performTokenRefresh();
  }

  // Shared implementation to avoid duplication between instance and static methods
  private static async performTokenRefresh(): Promise<string | null> {
    try {
      if (window.Clerk?.session) {
        const newToken = await window.Clerk.session.getToken({
          expirationBufferSeconds: 15,
        });
        if (newToken) {
          cachedAuthToken = newToken;
        } else {
          console.warn(
            "Clerk session exists but getToken() returned empty - clearing stale token",
          );
          cachedAuthToken = null;
        }
        return cachedAuthToken;
      }

      if (window.Clerk?.user) {
        console.warn(
          "Clerk user exists but session is missing - user will be treated as anonymous",
        );
        console.warn("Attempting to recover session...");

        try {
          await window.Clerk.load?.();
          if (window.Clerk?.session) {
            const recoveredToken = await window.Clerk.session.getToken({
              expirationBufferSeconds: 15,
            });
            if (recoveredToken) {
              cachedAuthToken = recoveredToken;
              console.log("Session recovered successfully");
              return cachedAuthToken;
            }
          }
          console.warn("Session recovery failed - session still missing");
        } catch (recoveryErr: unknown) {
          console.error("Failed to recover Clerk session:", recoveryErr);
        }
      } else {
        cachedAuthToken = null;
      }

      return cachedAuthToken;
    } catch (e: unknown) {
      console.error("Failed to refresh auth token:", e);
      if (cachedAuthToken && !isTokenExpired(cachedAuthToken)) {
        return cachedAuthToken;
      }
      return null;
    }
  }

  // Track if we're currently refreshing token to prevent duplicate retries
  private static isRefreshingToken = false;
  // Queue of pending retry requests
  private static pendingRetries: Array<{
    element: HTMLElement;
    originalTrigger: string;
  }> = [];

  private static async handleTokenRefreshAndRetry() {
    ClerkAuth.isRefreshingToken = true;
    console.log("Refreshing auth token before retrying HTMX requests...");

    try {
      const newToken = await ClerkAuth.performTokenRefresh();

      if (newToken) {
        console.log(
          "Token refreshed successfully, retrying",
          ClerkAuth.pendingRetries.length,
          "requests",
        );

        const retries = [...ClerkAuth.pendingRetries];
        ClerkAuth.pendingRetries = [];

        for (const request of retries) {
          if (window.htmx) {
            const method = request.element.getAttribute("hx-get")
              ? "GET"
              : "POST";
            const url =
              request.element.getAttribute("hx-get") ||
              request.element.getAttribute("hx-post") ||
              "";
            const target = request.element.getAttribute("hx-target") || "";
            const swap = request.element.getAttribute("hx-swap") || "innerHTML";

            window.htmx.ajax(method, url, {
              source: request.element,
              target: target,
              swap: swap,
            });
          }
        }
      } else {
        console.error("Failed to refresh token - requests will remain blocked");
        ClerkAuth.pendingRetries = [];
        document.body.dispatchEvent(
          new CustomEvent("htmx:authRefreshFailed", {
            detail: { message: "Authentication failed. Please try again." },
          }),
        );
      }
    } catch (err: unknown) {
      console.error("Error during token refresh for HTMX retry:", err);
      ClerkAuth.pendingRetries = [];
    } finally {
      ClerkAuth.isRefreshingToken = false;
    }
  }

  private registerHtmxAuthHeader() {
    if (ClerkAuth.htmxAuthHeaderRegistered) return;
    ClerkAuth.htmxAuthHeaderRegistered = true;

    document.body.addEventListener("htmx:configRequest", (event) => {
      const htmxEvent = event as HtmxConfigRequestEvent;

      const tokenExpired = cachedAuthToken
        ? isTokenExpired(cachedAuthToken)
        : true;

      if (cachedAuthToken && !tokenExpired) {
        htmxEvent.detail.headers["Authorization"] = `Bearer ${cachedAuthToken}`;
      } else if (window.Clerk?.user) {
        if (tokenExpired && cachedAuthToken) {
          console.warn(
            "HTMX request: Auth token expired. Cancelling request to refresh token and retry...",
          );
        } else {
          console.warn(
            "HTMX request: User is logged in but no auth token available. " +
              "Cancelling request to refresh token and retry...",
          );
        }

        // Synchronously prevent this request
        event.preventDefault();

        // Store the element that triggered this request for retry
        const triggerElement = htmxEvent.detail.elt as HTMLElement;

        // Add to pending retries if not already there
        const alreadyPending = ClerkAuth.pendingRetries.some(
          (r) => r.element === triggerElement,
        );
        if (!alreadyPending) {
          ClerkAuth.pendingRetries.push({
            element: triggerElement,
            originalTrigger: triggerElement.getAttribute("hx-trigger") || "",
          });
        }

        // Handle async refresh separately
        if (!ClerkAuth.isRefreshingToken) {
          ClerkAuth.handleTokenRefreshAndRetry();
        }
      }
    });
  }

  private updateAuthUI() {
    const user = window.Clerk?.user;
    const desktopContainer = document.getElementById("desktop-auth-container");
    const mobileContainer = document.getElementById("mobile-auth-container");

    // Clear containers
    if (desktopContainer) desktopContainer.innerHTML = "";
    if (mobileContainer) mobileContainer.innerHTML = "";

    if (user) {
      // Render User Button
      this.mountUserButton(desktopContainer, "desktop");
      this.mountUserButton(mobileContainer, "mobile");
    } else {
      // Render Sign In and Sign Up Buttons
      this.renderAuthButtons(desktopContainer, "desktop");
      this.renderAuthButtons(mobileContainer, "mobile");

      // Try to open Google One Tap
      this.openGoogleOneTap();
    }
  }

  private openGoogleOneTap() {
    try {
      if (window.Clerk?.openGoogleOneTap) {
        const params = {
          cancelOnTapOutside: false,
          itpSupport: true,
          fedCmSupport: true,
        };
        window.Clerk.openGoogleOneTap(params);
      }
    } catch (err: unknown) {
      console.error("Error opening Google One Tap:", err);
    }
  }

  private mountUserButton(
    container: HTMLElement | null,
    type: "desktop" | "mobile",
  ) {
    if (!container) return;

    const el = document.createElement("div");
    el.id = `clerk-user-button-${type}`;
    el.className =
      type === "desktop"
        ? "auth-user-button-root flex h-8 w-8 shrink-0 items-center justify-center leading-none"
        : "auth-user-button-root";
    container.appendChild(el);

    try {
      window.Clerk?.mountUserButton(el, {
        appearance: {
          elements: {
            userButtonTrigger:
              "h-8 w-8 rounded-full p-0 leading-none hover:bg-transparent focus:bg-transparent active:bg-transparent focus-visible:bg-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-300 focus-visible:ring-offset-2 focus-visible:ring-offset-sky-100",
            userButtonAvatarBox: "h-8 w-8 rounded-full overflow-hidden",
            userButtonBox: "h-8 w-8 rounded-full overflow-hidden",
          },
        },
      });
    } catch (err: unknown) {
      console.error(`Error mounting ${type} user button:`, err);
    }
  }

  private renderAuthButtons(
    container: HTMLElement | null,
    type: "desktop" | "mobile",
  ) {
    if (!container) return;

    const signInBtn = document.createElement("div");
    if (type === "desktop") {
      signInBtn.id = "clerk-sign-in-button-desktop";
      signInBtn.className = `${SIGN_IN_CLASS} ${DESKTOP_AUTH_BUTTON_SIZE_CLASS}`;
    } else {
      signInBtn.id = "clerk-sign-in-button-mobile";
      signInBtn.className = `${SIGN_IN_CLASS} ${MOBILE_AUTH_BUTTON_SIZE_CLASS} w-full text-center mb-2`;
    }
    signInBtn.textContent = "Sign In";
    signInBtn.addEventListener("click", (e) => {
      e.preventDefault();
      ClerkHelpers.openSignIn();
    });
    container.appendChild(signInBtn);

    const signUpBtn = document.createElement("div");
    if (type === "desktop") {
      signUpBtn.id = "clerk-sign-up-button-desktop";
      signUpBtn.className = `${SIGN_UP_CLASS} ${DESKTOP_AUTH_BUTTON_SIZE_CLASS} ml-2`;
    } else {
      signUpBtn.id = "clerk-sign-up-button-mobile";
      signUpBtn.className = `${SIGN_UP_CLASS} ${MOBILE_AUTH_BUTTON_SIZE_CLASS} w-full text-center mb-2`;
    }
    signUpBtn.textContent = "Sign Up";
    signUpBtn.addEventListener("click", (e) => {
      e.preventDefault();
      ClerkHelpers.openSignUp();
    });
    container.appendChild(signUpBtn);
  }

  private renderFallbackAuth() {
    const desktopContainer = document.getElementById("desktop-auth-container");
    const mobileContainer = document.getElementById("mobile-auth-container");

    const redirectUrl = encodeURIComponent(window.location.href);

    const createFallbackLinks = (type: "desktop" | "mobile") => {
      const signInLink = document.createElement("a");
      signInLink.href =
        "https://accounts.classifast.com/sign-in?redirect_url=" + redirectUrl;
      signInLink.textContent = "Sign In";

      const signUpLink = document.createElement("a");
      signUpLink.href =
        "https://accounts.classifast.com/sign-up?redirect_url=" + redirectUrl;
      signUpLink.textContent = "Sign Up";

      if (type === "desktop") {
        signInLink.className = `${SIGN_IN_CLASS} ${DESKTOP_AUTH_BUTTON_SIZE_CLASS}`;
        signUpLink.className = `${SIGN_UP_CLASS} ${DESKTOP_AUTH_BUTTON_SIZE_CLASS} ml-2`;
      } else {
        signInLink.className = `${SIGN_IN_CLASS} ${MOBILE_AUTH_BUTTON_SIZE_CLASS} w-full text-center mb-2 block`;
        signUpLink.className = `${SIGN_UP_CLASS} ${MOBILE_AUTH_BUTTON_SIZE_CLASS} w-full text-center mb-2 block`;
      }

      return [signInLink, signUpLink];
    };

    if (desktopContainer) {
      desktopContainer.innerHTML = "";
      createFallbackLinks("desktop").forEach((link) =>
        desktopContainer.appendChild(link),
      );
    }
    if (mobileContainer) {
      mobileContainer.innerHTML = "";
      createFallbackLinks("mobile").forEach((link) =>
        mobileContainer.appendChild(link),
      );
    }

    // Signal auth ready even without Clerk (user is anonymous, fire only once)
    signalAuthReady();
  }

  // Public method to get current auth token
  static getCachedAuthToken(): string | null {
    if (cachedAuthToken && isTokenExpired(cachedAuthToken)) {
      return null;
    }
    return cachedAuthToken;
  }

  // Public method to refresh auth token
  static async refreshAuthToken(): Promise<string | null> {
    return await ClerkAuth.performTokenRefresh();
  }
}

// Result copy functionality with tooltip
export class ResultCopier {
  constructor() {
    this.init();
  }

  private init() {
    // Expose global function for inline HTML onclick handlers
    window.copyOriginalId = (text: string, buttonElement: HTMLButtonElement) =>
      this.copy(text, buttonElement);
  }

  private copy(text: string, buttonElement: HTMLButtonElement) {
    if (!navigator.clipboard) {
      this.fallbackCopy(text, buttonElement);
      return;
    }

    navigator.clipboard
      .writeText(text)
      .then(() => {
        this.showTooltip(buttonElement, "Copied!");
      })
      .catch((err: unknown) => {
        console.error("Async: Could not copy text: ", err);
        this.showTooltip(buttonElement, "Copy failed");
      });
  }

  private fallbackCopy(text: string, buttonElement: HTMLButtonElement) {
    // Fallback for older browsers or insecure contexts (e.g. http)
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      document.execCommand("copy");
      this.showTooltip(buttonElement, "Copied!");
    } catch (err: unknown) {
      console.error("Fallback: Oops, unable to copy", err);
      this.showTooltip(buttonElement, "Copy failed");
    }
    document.body.removeChild(textArea);
  }

  private showTooltip(buttonElement: HTMLButtonElement, message: string) {
    const tooltip = document.createElement("span");
    tooltip.textContent = message;
    // Basic styling for the tooltip
    tooltip.style.position = "absolute";
    tooltip.style.backgroundColor = "black";
    tooltip.style.color = "white";
    tooltip.style.padding = "4px 8px";
    tooltip.style.borderRadius = "4px";
    tooltip.style.fontSize = "1.125rem";
    tooltip.style.zIndex = "1000";
    tooltip.style.textAlign = "center";

    // Append to body to avoid clipping issues and for correct initial dimension calculation
    document.body.appendChild(tooltip);

    const buttonRect = buttonElement.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();

    // Position above the button, centered, with scroll offset
    let top = buttonRect.top + window.scrollY - tooltipRect.height - 5;
    let left =
      buttonRect.left +
      window.scrollX +
      buttonRect.width / 2 -
      tooltipRect.width / 2;

    // Adjust if tooltip goes off-screen (viewport relative checks)
    if (buttonRect.top - tooltipRect.height - 5 < 0) {
      top = buttonRect.bottom + window.scrollY + 5;
    }
    if (left - window.scrollX < 0) {
      left = window.scrollX;
    }
    if (left - window.scrollX + tooltipRect.width > window.innerWidth) {
      left = window.scrollX + window.innerWidth - tooltipRect.width;
    }

    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;

    buttonElement.disabled = true;

    setTimeout(() => {
      if (tooltip.parentNode) {
        tooltip.parentNode.removeChild(tooltip);
      }
      buttonElement.disabled = false;
    }, 500);
  }
}

// Initialize common functionality when DOM is ready
export function initCommon(): void {
  new MobileMenu();
  new ClerkAuth();
  new TextareaEnhancer("product_description_area");
  new ResultCopier();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initCommon);
} else {
  initCommon();
}

// Expose ShareLink globally for inline onclick handlers
window.ShareLink = ShareLink;
