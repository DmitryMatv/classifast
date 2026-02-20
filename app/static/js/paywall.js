(() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __moduleCache = /* @__PURE__ */ new WeakMap;
  var __toCommonJS = (from) => {
    var entry = __moduleCache.get(from), desc;
    if (entry)
      return entry;
    entry = __defProp({}, "__esModule", { value: true });
    if (from && typeof from === "object" || typeof from === "function")
      __getOwnPropNames(from).map((key) => !__hasOwnProp.call(entry, key) && __defProp(entry, key, {
        get: () => from[key],
        enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable
      }));
    __moduleCache.set(from, entry);
    return entry;
  };
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, {
        get: all[name],
        enumerable: true,
        configurable: true,
        set: (newValue) => all[name] = () => newValue
      });
  };

  // app/assets/ts/clerk-helpers.ts
  class ClerkHelpers {
    static openSignIn(fallbackButtonId) {
      if (window.Clerk?.openSignIn) {
        window.Clerk.openSignIn({ redirectUrl: window.location.href });
      } else {
        const redirectUrl = encodeURIComponent(window.location.href);
        window.location.href = `https://accounts.classifast.com/sign-in?redirect_url=${redirectUrl}`;
      }
    }
    static openSignUp() {
      if (window.Clerk?.openSignUp) {
        window.Clerk.openSignUp({ redirectUrl: window.location.href });
      } else {
        const redirectUrl = encodeURIComponent(window.location.href);
        window.location.href = `https://accounts.classifast.com/sign-up?redirect_url=${redirectUrl}`;
      }
    }
    static createAuthErrorMessage() {
      const errorDiv = document.createElement("div");
      errorDiv.className = "bg-amber-50 border border-amber-200 rounded-lg p-3 mt-3 text-center";
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
    static showAuthErrorAndRedirect(containerId, action, fallbackUrl, delayMs = 2000) {
      const container = document.getElementById(containerId);
      if (container) {
        container.appendChild(this.createAuthErrorMessage());
      }
      const targetUrl = fallbackUrl || `https://accounts.classifast.com/${action}`;
      setTimeout(() => {
        try {
          const url = new URL(targetUrl, window.location.origin);
          url.searchParams.set("redirect_url", window.location.href);
          window.location.href = url.toString();
        } catch (err) {
          console.error("Error parsing URL, using fallback:", err);
          const separator = targetUrl.includes("?") ? "&" : "?";
          window.location.href = `${targetUrl}${separator}redirect_url=${encodeURIComponent(window.location.href)}`;
        }
      }, delayMs);
    }
    static submitForm(selector = "form[hx-get]") {
      const form = document.querySelector(selector);
      if (form && typeof form.requestSubmit === "function") {
        form.requestSubmit();
        return true;
      }
      return false;
    }
  }

  // app/assets/ts/paywall.ts
  if (!window.__paywallScriptParsed) {
    let initPaywall = function() {
      if (window.__paywallInitialized) {
        return;
      }
      window.__paywallInitialized = true;
      const paywallWarning = document.getElementById("paywall-warning");
      const paywallButtons = document.getElementById("paywall-buttons");
      if (!paywallWarning || !paywallButtons) {
        return;
      }
      new PaywallManager;
      const upgradeButton = document.getElementById("upgrade-button");
      if (upgradeButton) {
        new CheckoutManager;
      }
    };
    window.__paywallScriptParsed = true;
    const PAYWALL_PRODUCT_ID = "e157e32f-e91c-4d51-af66-0c2eb3b23d71";

    class PaywallManager {
      wasSignedIn = false;
      constructor() {
        this.init();
      }
      init() {
        this.setupRetryButton();
        this.setupClerkListener();
        this.setupAuthButtons();
      }
      submitClassificationForm() {
        ClerkHelpers.submitForm("form[hx-get]");
      }
      setupRetryButton() {
        const retryButton = document.getElementById("retry-button");
        if (retryButton) {
          const newRetryButton = retryButton.cloneNode(true);
          retryButton.parentNode?.replaceChild(newRetryButton, retryButton);
          newRetryButton.addEventListener("click", () => this.submitClassificationForm());
        }
      }
      setupClerkListener() {
        if (window.__paywallClerkListenerRegistered) {
          return;
        }
        window.__paywallClerkListenerRegistered = true;
        this.wasSignedIn = !!(window.Clerk && window.Clerk.user);
        if (window.Clerk?.addListener) {
          window.Clerk.addListener((resources) => {
            if (resources.user && !this.wasSignedIn) {
              this.wasSignedIn = true;
              this.submitClassificationForm();
            }
            if (!resources.user) {
              this.wasSignedIn = false;
            }
          });
        }
      }
      setupAuthButtons() {
        const signinButton = document.getElementById("signin-button");
        const signupButton = document.getElementById("signup-button");
        if (signinButton) {
          const newSigninButton = signinButton.cloneNode(true);
          signinButton.parentNode?.replaceChild(newSigninButton, signinButton);
          newSigninButton.addEventListener("click", (e) => {
            e.preventDefault();
            if (window.Clerk?.openSignIn) {
              window.Clerk.openSignIn({ redirectUrl: window.location.href });
            } else {
              const fallbackUrl = newSigninButton.dataset.fallbackUrl;
              ClerkHelpers.showAuthErrorAndRedirect("paywall-buttons", "sign-in", fallbackUrl);
            }
          });
        }
        if (signupButton) {
          const newSignupButton = signupButton.cloneNode(true);
          signupButton.parentNode?.replaceChild(newSignupButton, signupButton);
          newSignupButton.addEventListener("click", (e) => {
            e.preventDefault();
            if (window.Clerk?.openSignUp) {
              window.Clerk.openSignUp({ redirectUrl: window.location.href });
            } else {
              const fallbackUrl = newSignupButton.dataset.fallbackUrl;
              ClerkHelpers.showAuthErrorAndRedirect("paywall-buttons", "sign-up", fallbackUrl);
            }
          });
        }
      }
    }

    class CheckoutManager {
      button = null;
      constructor() {
        this.init();
      }
      init() {
        this.setupUpgradeButton();
      }
      setupUpgradeButton() {
        const upgradeButton = document.getElementById("upgrade-button");
        if (!upgradeButton)
          return;
        const newUpgradeButton = upgradeButton.cloneNode(true);
        upgradeButton.parentNode?.replaceChild(newUpgradeButton, upgradeButton);
        this.button = newUpgradeButton;
        newUpgradeButton.addEventListener("click", async (e) => {
          e.preventDefault();
          await this.handleUpgrade(newUpgradeButton);
        });
      }
      async handleUpgrade(button) {
        if (!window.Clerk?.session) {
          console.error("Clerk not available");
          this.showErrorState(button);
          return;
        }
        try {
          this.showLoadingState(button);
          const controller = new AbortController;
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
                Authorization: `Bearer ${token}`
              },
              signal: controller.signal,
              body: JSON.stringify({
                product_id: PAYWALL_PRODUCT_ID,
                return_url: window.location.href
              })
            });
            if (!response.ok)
              throw new Error("Checkout creation failed");
            const data = await response.json();
            if (data.url) {
              window.location.href = data.url;
            } else {
              throw new Error("No checkout URL returned");
            }
          } finally {
            clearTimeout(timeoutId);
          }
        } catch (err) {
          this.handleError(err, button);
        }
      }
      showLoadingState(button) {
        button.disabled = true;
        button.innerHTML = `
      <svg class="w-4 h-4 mr-2 inline animate-spin" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>Preparing...
    `;
      }
      showErrorState(button) {
        button.innerHTML = `
      <svg class="w-4 h-4 mr-2 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>Error - Try again
    `;
        button.disabled = false;
      }
      handleError(err, button) {
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
    document.addEventListener("DOMContentLoaded", () => {
      initPaywall();
    });
    document.body.addEventListener("htmx:afterSwap", (evt) => {
      const htmxEvent = evt;
      if (htmxEvent.detail.target.id === "results-container") {
        window.__paywallInitialized = false;
        setTimeout(initPaywall, 0);
      }
    });
    if (document.readyState !== "loading") {
      setTimeout(initPaywall, 0);
    }
  }
})();
