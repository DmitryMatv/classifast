import "./types/globals";
import { ClerkHelpers } from "./clerk-helpers";

// Shared TypeScript functionality for Classifast application

const SIGN_IN_CLASS =
  "inline-flex shrink-0 items-center justify-center whitespace-nowrap bg-sky-50 text-sky-700 hover:bg-sky-100 active:bg-sky-100 active:scale-95 rounded transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded";
const SIGN_UP_CLASS =
  "inline-flex shrink-0 items-center justify-center whitespace-nowrap bg-sky-700 hover:bg-sky-800 active:bg-sky-900 active:scale-95 text-white rounded transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded";
const DESKTOP_AUTH_BUTTON_SIZE_CLASS = "h-9 px-4 leading-none";
const MOBILE_AUTH_BUTTON_SIZE_CLASS = "min-h-9 px-4 py-2 leading-none";
const CLERK_SCRIPT_READINESS_TIMEOUT_MS = 10000;
const CLERK_LOAD_TIMEOUT_MS = 10000;
const INITIAL_TOKEN_REFRESH_TIMEOUT_MS = 10000;
const DEFAULT_EXAMPLE_CLEAR_DELAY_MS = 200;

// Global error handlers
window.addEventListener("error", (event) => {
  console.error("Global error:", event.error);
});

window.addEventListener("unhandledrejection", (event) => {
  console.error("Unhandled promise rejection:", event.reason);
});

// Mobile menu functionality
export class MobileMenu {
  private button: HTMLButtonElement | null = null;
  private menu: HTMLElement | null = null;
  private hamburger: HTMLElement | null = null;

  constructor() {
    this.init();
  }

  private init() {
    this.button = document.getElementById(
      "mobile-menu-button",
    ) as HTMLButtonElement | null;
    this.menu = document.getElementById("mobile-menu");
    this.hamburger = this.button?.matches(".hamburger")
      ? this.button
      : (this.button?.querySelector(".hamburger") ??
        document.querySelector(".hamburger"));

    if (!this.button || !this.menu || !this.hamburger) return;

    this.button.setAttribute("aria-controls", this.menu.id);
    this.button.addEventListener("click", (event) => {
      event.stopPropagation();
      this.toggle();
    });

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
  private defaultExampleCleared = false;
  private defaultExampleClearTimeoutId: number | null = null;

  constructor(textareaId: string) {
    this.textarea = document.getElementById(
      textareaId,
    ) as HTMLTextAreaElement | null;
    if (this.textarea) {
      this.init();
    }
  }

  private init() {
    this.setupDefaultExampleClear();

    this.textarea?.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.submitForm();
      }
    });
  }

  private isDefaultExamplePrefill(): boolean {
    const form = this.textarea?.closest("form");
    return form?.dataset["defaultExamplePrefill"] === "true";
  }

  private setupDefaultExampleClear() {
    if (!this.textarea || !this.isDefaultExamplePrefill()) {
      return;
    }

    const initialValue = this.textarea.value;
    if (!initialValue) {
      return;
    }

    const clearDefaultExampleTimeout = () => {
      if (this.defaultExampleClearTimeoutId === null) {
        return;
      }

      window.clearTimeout(this.defaultExampleClearTimeoutId);
      this.defaultExampleClearTimeoutId = null;
    };

    this.textarea.addEventListener("input", clearDefaultExampleTimeout, {
      once: true,
    });

    this.defaultExampleClearTimeoutId = window.setTimeout(() => {
      this.defaultExampleClearTimeoutId = null;

      if (
        !this.textarea ||
        this.defaultExampleCleared ||
        this.textarea.value !== initialValue
      ) {
        return;
      }

      this.defaultExampleCleared = true;
      this.textarea.value = "";
      this.textarea.defaultValue = "";
      this.textarea.textContent = "";
    }, DEFAULT_EXAMPLE_CLEAR_DELAY_MS);
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
  window.__authReady = true;

  if (!authReadyFired) {
    authReadyFired = true;
    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
  }
}

// Simple Clerk Authentication using official SDK patterns
export class ClerkAuth {
  private static htmxAuthHeaderRegistered = false;
  private clerkStarted = false;
  private hasAttemptedGoogleOneTap = false;
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
      await this.withTimeout(
        ClerkAuth.loadClerk(),
        CLERK_LOAD_TIMEOUT_MS,
        "Timed out waiting for Clerk.load()",
      );

      // Check if returning from successful checkout - clean up URL params
      const isCheckoutSuccess =
        new URLSearchParams(window.location.search).get("checkout") ===
        "success";

      if (isCheckoutSuccess) {
        this.cleanupCheckoutTokens();
        // Backend verifies tier via Redis-cached Clerk API - no frontend retries needed
      }

      // Initial token cache and UI render
      await this.withTimeout(
        this.refreshAuthToken(),
        INITIAL_TOKEN_REFRESH_TIMEOUT_MS,
        "Timed out waiting for initial Clerk token refresh",
      );

      // Only register the HTMX auth hook after the initial auth refresh settles.
      // If bootstrap falls back to anonymous mode, requests should not be cancelled.
      this.registerHtmxAuthHeader();

      this.updateAuthUI();

      // Signal that auth is ready for auto-classification (fire only once).
      // Metered direct-link autoloads wait for this so the first HTMX request
      // can include an authenticated Clerk token when available.
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

  private static async loadClerk(): Promise<void> {
    const clerk = window.Clerk;
    if (!clerk?.load) {
      throw new Error("Clerk unavailable");
    }

    const ClerkUI = await ClerkAuth.waitForClerkUiConstructor();
    await clerk.load({
      ui: {
        ClerkUI,
      },
    });
  }

  private async withTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number,
    errorMessage: string,
  ): Promise<T> {
    let timeoutId: number | null = null;

    try {
      return await Promise.race([
        promise,
        new Promise<T>((_, reject) => {
          timeoutId = window.setTimeout(() => {
            reject(new Error(errorMessage));
          }, timeoutMs);
        }),
      ]);
    } finally {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
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

    if (window.__clerkScriptFailed) {
      console.error("Clerk script failed to load");
      this.renderFallbackAuth();
      return;
    }

    let settled = false;
    let rejectOnScriptError: ((reason?: unknown) => void) | null = null;

    const cleanup = () => {
      script.removeEventListener("load", onScriptLoaded);
      script.removeEventListener("error", onScriptError);
      rejectOnScriptError = null;
    };

    const settleWithFallback = (message: string) => {
      if (settled) return;
      settled = true;
      cleanup();
      console.error(message);
      this.renderFallbackAuth();
    };

    const onScriptLoaded = () => {
      // Readiness is detected by waitForClerkInstance polling; this handler
      // exists only to be cleaned up and to guard against duplicate handling.
      if (settled) return;
    };

    const onScriptError = () => {
      const reject = rejectOnScriptError;
      window.__clerkScriptFailed = true;
      settleWithFallback("Clerk script failed to load");
      reject?.(new Error("Clerk script failed to load"));
    };

    script.addEventListener("load", onScriptLoaded, { once: true });
    script.addEventListener("error", onScriptError, { once: true });

    let clerk: ClerkInstance | null = null;
    try {
      clerk = await Promise.race([
        this.waitForClerkInstance(CLERK_SCRIPT_READINESS_TIMEOUT_MS),
        new Promise<ClerkInstance | null>((_, reject) => {
          rejectOnScriptError = reject;
        }),
      ]);
    } catch {
      return;
    }

    if (settled) {
      return;
    }

    if (clerk) {
      settled = true;
      cleanup();
      await this.startClerk();
      return;
    }

    settleWithFallback("Timed out waiting for Clerk script readiness");
  }

  private async waitForClerkInstance(
    timeoutMs = CLERK_SCRIPT_READINESS_TIMEOUT_MS,
  ): Promise<ClerkInstance | null> {
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      if (window.Clerk) return window.Clerk;
      await new Promise((resolve) => window.setTimeout(resolve, 50));
    }

    return window.Clerk ?? null;
  }

  private static async waitForClerkUiConstructor(
    timeoutMs = CLERK_SCRIPT_READINESS_TIMEOUT_MS,
  ): Promise<NonNullable<Window["__internal_ClerkUICtor"]>> {
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      const ClerkUI = window.__internal_ClerkUICtor;
      if (ClerkUI) return ClerkUI;
      await new Promise((resolve) => window.setTimeout(resolve, 50));
    }

    throw new Error("Clerk UI bundle unavailable");
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
          await ClerkAuth.loadClerk();
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

  private shouldOpenGoogleOneTap(): boolean {
    if (this.hasAttemptedGoogleOneTap || window.Clerk?.user) {
      return false;
    }

    try {
      return window.top === window.self;
    } catch {
      return false;
    }
  }

  private openGoogleOneTap() {
    try {
      if (this.shouldOpenGoogleOneTap() && window.Clerk?.openGoogleOneTap) {
        this.hasAttemptedGoogleOneTap = true;
        const params = {
          cancelOnTapOutside: false,
          itpSupport: true,
          // GIS/FedCM is the intended path here. The remaining FedCM migration
          // warning is emitted by Clerk's bundled One Tap wrapper, not by our code.
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
        ? "auth-user-button-root flex h-9 w-9 shrink-0 items-center justify-center leading-none"
        : "auth-user-button-root";
    container.appendChild(el);

    try {
      window.Clerk?.mountUserButton(el, {
        appearance: {
          elements: {
            userButtonTrigger:
              "inline-flex h-9 w-9 items-center justify-center overflow-hidden rounded-full border-0 bg-transparent p-0 leading-none shadow-none outline-none hover:bg-transparent focus:bg-transparent focus:shadow-none focus:outline-none active:bg-transparent active:shadow-none focus-visible:bg-transparent focus-visible:shadow-none focus-visible:outline-none focus-visible:ring-0",
            userButtonAvatarBox:
              "h-9 w-9 rounded-full overflow-hidden border-0 bg-transparent p-0 shadow-none",
            userButtonBox:
              "h-9 w-9 rounded-full overflow-hidden border-0 bg-transparent p-0 shadow-none",
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

    const signInBtn = document.createElement("button");
    signInBtn.type = "button";
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

    const signUpBtn = document.createElement("button");
    signUpBtn.type = "button";
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
    this.cleanupCheckoutTokens();

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

    // Signal auth ready even without Clerk (user is anonymous, fire only once).
    // Direct-link autoloads rely on this fallback so they do not hang forever
    // if Clerk cannot bootstrap on the page.
    signalAuthReady();
  }

  private cleanupCheckoutTokens() {
    const url = new URL(window.location.href);
    const hadCheckoutParams =
      url.searchParams.has("checkout_token") ||
      url.searchParams.has("customer_session_token");

    if (!hadCheckoutParams) {
      return;
    }

    url.searchParams.delete("checkout_token");
    url.searchParams.delete("customer_session_token");

    const cleanUrl = `${url.pathname}${url.search}${url.hash}`;
    window.history.replaceState({}, "", cleanUrl);
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
        this.fallbackCopy(text, buttonElement);
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
      const didCopy = document.execCommand("copy");
      this.showTooltip(buttonElement, didCopy ? "Copied!" : "Copy failed");
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
  if (document.body.dataset["commonInitialized"] === "true") {
    return;
  }

  document.body.dataset["commonInitialized"] = "true";
  new MobileMenu();
  if (document.body.dataset["authUi"] !== "disabled") {
    new ClerkAuth();
  }
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
