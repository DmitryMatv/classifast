import "./types/globals";

// Shared TypeScript functionality for Classifast application

// Mobile menu functionality
class MobileMenu {
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
class ShareLink {
  static async copyShareableLink() {
    const url = window.location.href;
    const button = document.getElementById("share-button");

    try {
      await navigator.clipboard.writeText(url);
      this.showFeedback(button);
    } catch (err) {
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
    } catch (fallbackErr) {
      console.error("Fallback copy failed: ", fallbackErr);
    }

    document.body.removeChild(textArea);
  }
}

// Textarea enhanced functionality
class TextareaEnhancer {
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

// Toggle functionality for classifier description sections
class DescriptionToggle {
  constructor() {
    this.init();
  }

  private init() {
    const toggle = document.getElementById("description-toggle");
    const content = document.getElementById("description-content");
    const container = document.getElementById("description-container");

    if (!toggle || !content || !container) return;

    // Hide entire block if description empty
    const text = content.textContent ?? "";
    if (!text.trim()) {
      toggle.style.display = "none";
      container.style.display = "none";
      return;
    }

    this.setupToggle(toggle, content);
  }

  private setupToggle(toggle: HTMLElement, content: HTMLElement) {
    const logos = document.querySelectorAll<HTMLElement>(
      "[data-classifier-logo]",
    );
    const learnLabel =
      toggle.getAttribute("aria-label")?.replace(" button", "") ?? "Learn more";
    const showLessLabel = "Show less";

    // Initialize state: hidden
    content.style.display = "none";
    content.setAttribute("aria-hidden", "true");
    toggle.setAttribute("aria-expanded", "false");
    toggle.textContent = learnLabel;

    toggle.addEventListener("click", (e) => {
      e.preventDefault();
      const isHidden =
        content.style.display === "none" || content.style.display === "";

      if (isHidden) {
        this.showContent(content, toggle, logos, showLessLabel);
      } else {
        this.hideContent(content, toggle, logos, learnLabel);
      }
    });
  }

  private showContent(
    content: HTMLElement,
    toggle: HTMLElement,
    logos: NodeListOf<HTMLElement>,
    showLessLabel: string,
  ) {
    content.style.display = "block";
    content.setAttribute("aria-hidden", "false");
    toggle.setAttribute("aria-expanded", "true");
    toggle.textContent = showLessLabel;

    logos.forEach((logo) => {
      if (!logo.dataset.originalDisplay) {
        logo.dataset.originalDisplay = logo.style.display ?? "";
      }
      logo.style.display = "none";
    });
  }

  private hideContent(
    content: HTMLElement,
    toggle: HTMLElement,
    logos: NodeListOf<HTMLElement>,
    learnLabel: string,
  ) {
    content.style.display = "none";
    content.setAttribute("aria-hidden", "true");
    toggle.setAttribute("aria-expanded", "false");
    toggle.textContent = learnLabel;

    logos.forEach((logo) => {
      const original = logo.dataset.originalDisplay ?? "";
      logo.style.display = original;
    });
  }
}

// Cached auth token for synchronous HTMX header injection
let cachedAuthToken: string | null = null;

// Track if auth-ready event has been fired (fire only once on initial load)
let authReadyFired = false;

// Simple Clerk Authentication using official SDK patterns
class ClerkAuth {
  constructor() {
    this.init();
  }

  private async init() {
    if (window.Clerk) {
      await this.startClerk();
      return;
    }

    if (document.readyState === "complete") {
      this.renderFallbackAuth();
      return;
    }

    window.addEventListener("load", async () => {
      if (window.Clerk) {
        await this.startClerk();
      } else {
        console.error("Clerk script failed to load");
        this.renderFallbackAuth();
      }
    });
  }

  private async startClerk() {
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
      if (!authReadyFired) {
        authReadyFired = true;
        document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
      }

      // Refresh token every 50s (Clerk tokens expire in ~60s)
      // Needed for long sessions on single page
      setInterval(() => this.refreshAuthToken(), 50000);

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
    } catch (err) {
      console.error("Error initializing Clerk:", err);
      this.renderFallbackAuth();
    }
  }

  private async refreshAuthToken() {
    try {
      if (window.Clerk?.session) {
        cachedAuthToken = await window.Clerk.session.getToken();
        if (!cachedAuthToken) {
          console.warn("Clerk session exists but getToken() returned empty");
        }
      } else {
        cachedAuthToken = null;
        // Log when user exists but session doesn't - this is the problematic state
        if (window.Clerk?.user) {
          console.warn(
            "Clerk user exists but session is missing - user will be treated as anonymous",
          );
          console.warn("Attempting to recover session...");

          // Try to recover: reload Clerk to re-initialize session
          try {
            await window.Clerk.load?.();
            // Retry token retrieval after reload
            if (window.Clerk?.session) {
              cachedAuthToken = await window.Clerk.session.getToken();
              if (cachedAuthToken) {
                console.log("Session recovered successfully");
              }
            }
          } catch (recoveryErr) {
            console.error("Failed to recover Clerk session:", recoveryErr);
          }
        }
      }
    } catch (e) {
      console.error("Failed to refresh auth token:", e);
      cachedAuthToken = null;
    }
  }

  private registerHtmxAuthHeader() {
    document.body.addEventListener("htmx:configRequest", (event) => {
      const htmxEvent = event as HtmxConfigRequestEvent;
      if (cachedAuthToken) {
        htmxEvent.detail.headers["Authorization"] = `Bearer ${cachedAuthToken}`;
      } else if (window.Clerk?.user) {
        // If user exists but token is missing, log this diagnostic
        console.warn(
          "HTMX request: User is logged in but no auth token available - request will be treated as anonymous",
        );
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
      // Enable flex layout for desktop to align items
      if (desktopContainer) {
        desktopContainer.style.display = "flex";
        desktopContainer.style.alignItems = "center";
      }

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
    } catch (err) {
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
    if (type === "desktop") {
      el.className = "flex items-center";
    }
    container.appendChild(el);

    try {
      window.Clerk?.mountUserButton(el, {
        appearance: {
          elements: {
            userButtonAvatarBox: "w-8 h-8",
            userButtonBox: "h-8",
          },
        },
      });
    } catch (err) {
      console.error(`Error mounting ${type} user button:`, err);
    }
  }

  private renderAuthButtons(
    container: HTMLElement | null,
    type: "desktop" | "mobile",
  ) {
    if (!container) return;

    // Enable flex layout for desktop to show buttons side-by-side
    if (type === "desktop") {
      container.style.display = "flex";
      container.style.alignItems = "center";
    }

    const signInClass =
      "bg-sky-50 text-sky-700 hover:bg-sky-100 active:bg-sky-100 active:scale-95 px-4 py-1 rounded text-sm transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded";
    const signUpClass =
      "bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white px-4 py-1 rounded text-sm transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded";

    // Sign In button (outline style)
    const signInBtn = document.createElement("div");
    if (type === "desktop") {
      signInBtn.id = "clerk-sign-in-button-desktop";
      signInBtn.className = signInClass;
    } else {
      signInBtn.id = "clerk-sign-in-button-mobile";
      signInBtn.className = signInClass + " w-full text-center mb-2";
    }
    signInBtn.textContent = "Sign In";
    signInBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (window.Clerk?.openSignIn) {
        window.Clerk.openSignIn({ redirectUrl: window.location.href });
      } else {
        window.location.href =
          "https://accounts.classifast.com/sign-in?redirect_url=" +
          encodeURIComponent(window.location.href);
      }
    });
    container.appendChild(signInBtn);

    // Sign Up button (filled style)
    const signUpBtn = document.createElement("div");
    if (type === "desktop") {
      signUpBtn.id = "clerk-sign-up-button-desktop";
      signUpBtn.className = signUpClass + " ml-2";
    } else {
      signUpBtn.id = "clerk-sign-up-button-mobile";
      signUpBtn.className = signUpClass + " w-full text-center mb-2";
    }
    signUpBtn.textContent = "Sign Up";
    signUpBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (window.Clerk?.openSignUp) {
        window.Clerk.openSignUp({ redirectUrl: window.location.href });
      } else {
        window.location.href =
          "https://accounts.classifast.com/sign-up?redirect_url=" +
          encodeURIComponent(window.location.href);
      }
    });
    container.appendChild(signUpBtn);
  }

  private renderFallbackAuth() {
    const desktopContainer = document.getElementById("desktop-auth-container");
    const mobileContainer = document.getElementById("mobile-auth-container");

    const redirectUrl = encodeURIComponent(window.location.href);
    const signInClass =
      "bg-sky-50 text-sky-700 hover:bg-sky-100 active:bg-sky-100 active:scale-95 px-4 py-1 rounded text-sm transition-all duration-150 ease-in-out transform auth-loaded";
    const signUpClass =
      "bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white px-4 py-1 rounded text-sm transition-all duration-150 ease-in-out transform auth-loaded";

    // Enable flex layout for desktop
    if (desktopContainer) {
      desktopContainer.style.display = "flex";
      desktopContainer.style.alignItems = "center";
    }

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
        signInLink.className = signInClass;
        signUpLink.className = signUpClass + " ml-2";
      } else {
        signInLink.className = signInClass + " w-full text-center mb-2 block";
        signUpLink.className = signUpClass + " w-full text-center mb-2 block";
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
    if (!authReadyFired) {
      authReadyFired = true;
      document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    }
  }
}

// Result copy functionality with tooltip
class ResultCopier {
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
    } catch (err) {
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

// Initialize common functionality
document.addEventListener("DOMContentLoaded", () => {
  new MobileMenu();
  new DescriptionToggle();
  new ClerkAuth();
  new TextareaEnhancer("product_description_area");
  new ResultCopier();
});
