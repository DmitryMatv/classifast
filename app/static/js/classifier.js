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

  // app/assets/ts/common.ts
  var exports_common = {};
  __export(exports_common, {
    TextareaEnhancer: () => TextareaEnhancer,
    ShareLink: () => ShareLink,
    ResultCopier: () => ResultCopier,
    MobileMenu: () => MobileMenu,
    ClerkAuth: () => ClerkAuth
  });

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

  // app/assets/ts/common.ts
  var SIGN_IN_CLASS = "bg-sky-50 text-sky-700 hover:bg-sky-100 active:bg-sky-100 active:scale-95 px-4 py-1 rounded text-sm transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded";
  var SIGN_UP_CLASS = "bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white px-4 py-1 rounded text-sm transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded";
  window.addEventListener("error", (event) => {
    console.error("Global error:", event.error);
  });
  window.addEventListener("unhandledrejection", (event) => {
    console.error("Unhandled promise rejection:", event.reason);
  });

  class MobileMenu {
    button = null;
    menu = null;
    hamburger = null;
    constructor() {
      this.init();
    }
    init() {
      this.button = document.getElementById("mobile-menu-button");
      this.menu = document.getElementById("mobile-menu");
      this.hamburger = document.querySelector(".hamburger");
      if (!this.button || !this.menu || !this.hamburger)
        return;
      this.button.addEventListener("click", () => this.toggle());
      const links = this.menu.querySelectorAll("a");
      links.forEach((link) => {
        link.addEventListener("click", () => this.close());
      });
      document.addEventListener("click", (e) => {
        if (!this.menu?.contains(e.target) && !this.button?.contains(e.target)) {
          this.close();
        }
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && this.menu?.classList.contains("active")) {
          this.close();
          this.button?.focus();
        }
      });
    }
    toggle() {
      const isActive = this.menu?.classList.toggle("active") ?? false;
      this.hamburger?.classList.toggle("active");
      this.button?.setAttribute("aria-expanded", String(isActive));
    }
    close() {
      this.menu?.classList.remove("active");
      this.hamburger?.classList.remove("active");
      this.button?.setAttribute("aria-expanded", "false");
    }
  }

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
    static showFeedback(button) {
      if (!button)
        return;
      const originalText = button.innerHTML;
      button.innerHTML = "Copied!";
      button.classList.add("bg-green-600", "hover:bg-green-700");
      setTimeout(() => {
        button.innerHTML = originalText;
        button.classList.remove("bg-green-600", "hover:bg-green-700");
      }, 2000);
    }
    static fallbackCopy(url, button) {
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

  class TextareaEnhancer {
    textarea;
    constructor(textareaId) {
      this.textarea = document.getElementById(textareaId);
      if (this.textarea) {
        this.init();
      }
    }
    init() {
      this.textarea?.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          this.submitForm();
        }
      });
    }
    submitForm() {
      const form = this.textarea?.closest("form");
      if (form) {
        const submitBtn = form.querySelector('button[type="submit"]');
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
    setValue(value) {
      if (this.textarea) {
        this.textarea.value = value;
      }
    }
  }
  var cachedAuthToken = null;
  function isTokenExpired(token) {
    try {
      const parts = token.split(".");
      const payloadB64 = parts[1];
      if (parts.length !== 3 || !payloadB64)
        return true;
      const base64 = payloadB64.replace(/-/g, "+").replace(/_/g, "/");
      const padded = base64 + "==".slice(0, (4 - base64.length % 4) % 4);
      const payload = JSON.parse(atob(padded));
      if (!payload.exp)
        return true;
      const now = Math.floor(Date.now() / 1000);
      return payload.exp < now;
    } catch (err) {
      console.error("Error parsing JWT token:", err);
      return true;
    }
  }
  var authReadyFired = false;

  class ClerkAuth {
    static htmxAuthHeaderRegistered = false;
    constructor() {
      this.init();
    }
    async init() {
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
    async startClerk() {
      try {
        await window.Clerk?.load();
        this.registerHtmxAuthHeader();
        const urlParams = new URLSearchParams(window.location.search);
        const isCheckoutSuccess = urlParams.get("checkout") === "success";
        if (isCheckoutSuccess) {
          urlParams.delete("checkout");
          urlParams.delete("customer_session_token");
          const cleanUrl = urlParams.toString() ? `${window.location.pathname}?${urlParams.toString()}` : window.location.pathname;
          window.history.replaceState({}, "", cleanUrl);
        }
        await this.refreshAuthToken();
        this.updateAuthUI();
        if (!authReadyFired) {
          authReadyFired = true;
          document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
        }
        setInterval(() => this.refreshAuthToken(), 50000);
        if (!window.__clerkInteractionListenersRegistered) {
          window.__clerkInteractionListenersRegistered = true;
          let refreshPending = false;
          const refreshOnInteraction = () => {
            const tokenExpired = cachedAuthToken ? isTokenExpired(cachedAuthToken) : true;
            if (window.Clerk?.session && (!cachedAuthToken || tokenExpired) && !refreshPending) {
              refreshPending = true;
              console.log("Refreshing token on user interaction...");
              this.refreshAuthToken().finally(() => {
                refreshPending = false;
              });
            }
          };
          document.addEventListener("click", refreshOnInteraction, {
            passive: true
          });
          document.addEventListener("keydown", refreshOnInteraction, {
            passive: true
          });
        }
        if (window.Clerk?.addListener && !window.__clerkAuthListenerRegistered) {
          window.__clerkAuthListenerRegistered = true;
          window.Clerk.addListener(async () => {
            await this.refreshAuthToken();
            this.updateAuthUI();
          });
        }
      } catch (err) {
        console.error("Error initializing Clerk:", err);
        this.renderFallbackAuth();
      }
    }
    async refreshAuthToken() {
      await ClerkAuth.performTokenRefresh();
    }
    static async performTokenRefresh() {
      try {
        if (window.Clerk?.session) {
          const newToken = await window.Clerk.session.getToken({
            expirationBufferSeconds: 15
          });
          if (newToken) {
            cachedAuthToken = newToken;
          } else {
            console.warn("Clerk session exists but getToken() returned empty - clearing stale token");
            cachedAuthToken = null;
          }
          return cachedAuthToken;
        }
        if (window.Clerk?.user) {
          console.warn("Clerk user exists but session is missing - user will be treated as anonymous");
          console.warn("Attempting to recover session...");
          try {
            await window.Clerk.load?.();
            if (window.Clerk?.session) {
              const recoveredToken = await window.Clerk.session.getToken({
                expirationBufferSeconds: 15
              });
              if (recoveredToken) {
                cachedAuthToken = recoveredToken;
                console.log("Session recovered successfully");
                return cachedAuthToken;
              }
            }
            console.warn("Session recovery failed - session still missing");
          } catch (recoveryErr) {
            console.error("Failed to recover Clerk session:", recoveryErr);
          }
        } else {
          cachedAuthToken = null;
        }
        return cachedAuthToken;
      } catch (e) {
        console.error("Failed to refresh auth token:", e);
        if (cachedAuthToken && !isTokenExpired(cachedAuthToken)) {
          return cachedAuthToken;
        }
        return null;
      }
    }
    static isRefreshingToken = false;
    static pendingRetries = [];
    static async handleTokenRefreshAndRetry() {
      ClerkAuth.isRefreshingToken = true;
      console.log("Refreshing auth token before retrying HTMX requests...");
      try {
        const newToken = await ClerkAuth.performTokenRefresh();
        if (newToken) {
          console.log("Token refreshed successfully, retrying", ClerkAuth.pendingRetries.length, "requests");
          const retries = [...ClerkAuth.pendingRetries];
          ClerkAuth.pendingRetries = [];
          for (const request of retries) {
            if (window.htmx) {
              const method = request.element.getAttribute("hx-get") ? "GET" : "POST";
              const url = request.element.getAttribute("hx-get") || request.element.getAttribute("hx-post") || "";
              const target = request.element.getAttribute("hx-target") || "";
              const swap = request.element.getAttribute("hx-swap") || "innerHTML";
              window.htmx.ajax(method, url, {
                source: request.element,
                target,
                swap
              });
            }
          }
        } else {
          console.error("Failed to refresh token - requests will remain blocked");
          ClerkAuth.pendingRetries = [];
          document.body.dispatchEvent(new CustomEvent("htmx:authRefreshFailed", {
            detail: { message: "Authentication failed. Please try again." }
          }));
        }
      } catch (err) {
        console.error("Error during token refresh for HTMX retry:", err);
        ClerkAuth.pendingRetries = [];
      } finally {
        ClerkAuth.isRefreshingToken = false;
      }
    }
    registerHtmxAuthHeader() {
      if (ClerkAuth.htmxAuthHeaderRegistered)
        return;
      ClerkAuth.htmxAuthHeaderRegistered = true;
      document.body.addEventListener("htmx:configRequest", (event) => {
        const htmxEvent = event;
        const tokenExpired = cachedAuthToken ? isTokenExpired(cachedAuthToken) : true;
        if (cachedAuthToken && !tokenExpired) {
          htmxEvent.detail.headers["Authorization"] = `Bearer ${cachedAuthToken}`;
        } else if (window.Clerk?.user) {
          if (tokenExpired && cachedAuthToken) {
            console.warn("HTMX request: Auth token expired. Cancelling request to refresh token and retry...");
          } else {
            console.warn("HTMX request: User is logged in but no auth token available. " + "Cancelling request to refresh token and retry...");
          }
          event.preventDefault();
          const triggerElement = htmxEvent.detail.elt;
          const alreadyPending = ClerkAuth.pendingRetries.some((r) => r.element === triggerElement);
          if (!alreadyPending) {
            ClerkAuth.pendingRetries.push({
              element: triggerElement,
              originalTrigger: triggerElement.getAttribute("hx-trigger") || ""
            });
          }
          if (!ClerkAuth.isRefreshingToken) {
            ClerkAuth.handleTokenRefreshAndRetry();
          }
        }
      });
    }
    updateAuthUI() {
      const user = window.Clerk?.user;
      const desktopContainer = document.getElementById("desktop-auth-container");
      const mobileContainer = document.getElementById("mobile-auth-container");
      if (desktopContainer)
        desktopContainer.innerHTML = "";
      if (mobileContainer)
        mobileContainer.innerHTML = "";
      if (user) {
        if (desktopContainer) {
          desktopContainer.style.display = "flex";
          desktopContainer.style.alignItems = "center";
        }
        this.mountUserButton(desktopContainer, "desktop");
        this.mountUserButton(mobileContainer, "mobile");
      } else {
        this.renderAuthButtons(desktopContainer, "desktop");
        this.renderAuthButtons(mobileContainer, "mobile");
        this.openGoogleOneTap();
      }
    }
    openGoogleOneTap() {
      try {
        if (window.Clerk?.openGoogleOneTap) {
          const params = {
            cancelOnTapOutside: false,
            itpSupport: true,
            fedCmSupport: true
          };
          window.Clerk.openGoogleOneTap(params);
        }
      } catch (err) {
        console.error("Error opening Google One Tap:", err);
      }
    }
    mountUserButton(container, type) {
      if (!container)
        return;
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
              userButtonBox: "h-8"
            }
          }
        });
      } catch (err) {
        console.error(`Error mounting ${type} user button:`, err);
      }
    }
    renderAuthButtons(container, type) {
      if (!container)
        return;
      if (type === "desktop") {
        container.style.display = "flex";
        container.style.alignItems = "center";
      }
      const signInBtn = document.createElement("div");
      if (type === "desktop") {
        signInBtn.id = "clerk-sign-in-button-desktop";
        signInBtn.className = SIGN_IN_CLASS;
      } else {
        signInBtn.id = "clerk-sign-in-button-mobile";
        signInBtn.className = SIGN_IN_CLASS + " w-full text-center mb-2";
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
        signUpBtn.className = SIGN_UP_CLASS + " ml-2";
      } else {
        signUpBtn.id = "clerk-sign-up-button-mobile";
        signUpBtn.className = SIGN_UP_CLASS + " w-full text-center mb-2";
      }
      signUpBtn.textContent = "Sign Up";
      signUpBtn.addEventListener("click", (e) => {
        e.preventDefault();
        ClerkHelpers.openSignUp();
      });
      container.appendChild(signUpBtn);
    }
    renderFallbackAuth() {
      const desktopContainer = document.getElementById("desktop-auth-container");
      const mobileContainer = document.getElementById("mobile-auth-container");
      const redirectUrl = encodeURIComponent(window.location.href);
      if (desktopContainer) {
        desktopContainer.style.display = "flex";
        desktopContainer.style.alignItems = "center";
      }
      const createFallbackLinks = (type) => {
        const signInLink = document.createElement("a");
        signInLink.href = "https://accounts.classifast.com/sign-in?redirect_url=" + redirectUrl;
        signInLink.textContent = "Sign In";
        const signUpLink = document.createElement("a");
        signUpLink.href = "https://accounts.classifast.com/sign-up?redirect_url=" + redirectUrl;
        signUpLink.textContent = "Sign Up";
        if (type === "desktop") {
          signInLink.className = SIGN_IN_CLASS;
          signUpLink.className = SIGN_UP_CLASS + " ml-2";
        } else {
          signInLink.className = SIGN_IN_CLASS + " w-full text-center mb-2 block";
          signUpLink.className = SIGN_UP_CLASS + " w-full text-center mb-2 block";
        }
        return [signInLink, signUpLink];
      };
      if (desktopContainer) {
        desktopContainer.innerHTML = "";
        createFallbackLinks("desktop").forEach((link) => desktopContainer.appendChild(link));
      }
      if (mobileContainer) {
        mobileContainer.innerHTML = "";
        createFallbackLinks("mobile").forEach((link) => mobileContainer.appendChild(link));
      }
      if (!authReadyFired) {
        authReadyFired = true;
        document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
      }
    }
    static getCachedAuthToken() {
      if (cachedAuthToken && isTokenExpired(cachedAuthToken)) {
        return null;
      }
      return cachedAuthToken;
    }
    static async refreshAuthToken() {
      return await ClerkAuth.performTokenRefresh();
    }
  }

  class ResultCopier {
    constructor() {
      this.init();
    }
    init() {
      window.copyOriginalId = (text, buttonElement) => this.copy(text, buttonElement);
    }
    copy(text, buttonElement) {
      if (!navigator.clipboard) {
        this.fallbackCopy(text, buttonElement);
        return;
      }
      navigator.clipboard.writeText(text).then(() => {
        this.showTooltip(buttonElement, "Copied!");
      }).catch((err) => {
        console.error("Async: Could not copy text: ", err);
        this.showTooltip(buttonElement, "Copy failed");
      });
    }
    fallbackCopy(text, buttonElement) {
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
    showTooltip(buttonElement, message) {
      const tooltip = document.createElement("span");
      tooltip.textContent = message;
      tooltip.style.position = "absolute";
      tooltip.style.backgroundColor = "black";
      tooltip.style.color = "white";
      tooltip.style.padding = "4px 8px";
      tooltip.style.borderRadius = "4px";
      tooltip.style.fontSize = "1.125rem";
      tooltip.style.zIndex = "1000";
      tooltip.style.textAlign = "center";
      document.body.appendChild(tooltip);
      const buttonRect = buttonElement.getBoundingClientRect();
      const tooltipRect = tooltip.getBoundingClientRect();
      let top = buttonRect.top + window.scrollY - tooltipRect.height - 5;
      let left = buttonRect.left + window.scrollX + buttonRect.width / 2 - tooltipRect.width / 2;
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
  document.addEventListener("DOMContentLoaded", () => {
    new MobileMenu;
    new ClerkAuth;
    new TextareaEnhancer("product_description_area");
    new ResultCopier;
  });
  if (document.readyState === "complete" || document.readyState === "interactive") {
    new MobileMenu;
    new ClerkAuth;
    new TextareaEnhancer("product_description_area");
    new ResultCopier;
  }
  window.ShareLink = ShareLink;

  // app/assets/ts/classifier.ts
  class ClassifierPage {
    constructor() {
      this.init();
    }
    init() {
      this.setupTopKAutosubmit();
      this.setupHTMXListeners();
      this.setupDescriptionToggle();
    }
    setupTopKAutosubmit() {
      const topKSelector = document.getElementById("show_top_k_categories");
      const productDescriptionArea = document.getElementById("product_description_area");
      if (topKSelector && productDescriptionArea) {
        topKSelector.addEventListener("change", () => {
          if (productDescriptionArea.value.trim()) {
            this.triggerFormSubmission();
          }
        });
      }
    }
    triggerFormSubmission() {
      const form = document.querySelector("form[hx-get]");
      const submitBtn = form?.querySelector('button[type="submit"]');
      if (form) {
        if (submitBtn) {
          submitBtn.classList.add("active", "scale-95");
          setTimeout(() => {
            submitBtn.classList.remove("active", "scale-95");
          }, 150);
        }
        if (window.htmx) {
          window.htmx.trigger(form, "submit");
        }
      }
    }
    setupHTMXListeners() {
      document.body.addEventListener("htmx:afterRequest", (evt) => {
        const htmxEvent = evt;
        const indicator = document.getElementById("loading-indicator");
        if (indicator && htmxEvent.detail.target.id === "results-container") {
          indicator.classList.remove("htmx-request");
        }
      });
      document.body.addEventListener("htmx:afterSwap", (evt) => {
        const htmxEvent = evt;
        if (htmxEvent.detail.target.id === "results-container") {
          const resultsSection = document.getElementById("results-section");
          if (resultsSection) {
            resultsSection.classList.remove("hidden");
          }
          this.attachShareButtonListener();
        }
      });
      document.body.addEventListener("htmx:responseError", (evt) => {
        const htmxEvent = evt;
        if (htmxEvent.detail.xhr.status === 429) {
          if (htmxEvent.detail.target.id === "results-container") {
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
    attachShareButtonListener() {
      const shareButton = document.getElementById("share-button");
      if (shareButton) {
        const newButton = shareButton.cloneNode(true);
        shareButton.parentNode?.replaceChild(newButton, shareButton);
        newButton.addEventListener("click", () => {
          this.copyShareableLink();
        });
      }
    }
    setupDescriptionToggle() {
      const toggleButton = document.getElementById("description-toggle");
      const descriptionContent = document.getElementById("description-content");
      const container = document.getElementById("description-container");
      if (!toggleButton || !descriptionContent || !container)
        return;
      const text = descriptionContent.textContent ?? "";
      if (!text.trim()) {
        toggleButton.style.display = "none";
        container.style.display = "none";
        return;
      }
      const classifierType = toggleButton.getAttribute("data-classifier-type") || "";
      const learnMoreText = classifierType ? `Learn more about ${classifierType}` : "Learn more";
      const isExpanded = toggleButton.getAttribute("aria-expanded") === "true";
      descriptionContent.style.display = isExpanded ? "block" : "none";
      descriptionContent.setAttribute("aria-hidden", String(!isExpanded));
      toggleButton.textContent = isExpanded ? "Show less" : learnMoreText;
      const initialLogoElements = document.querySelectorAll('[data-classifier-logo="true"]');
      initialLogoElements.forEach((logo) => {
        logo.style.display = isExpanded ? "none" : "";
      });
      toggleButton.addEventListener("click", () => {
        const currentlyExpanded = toggleButton.getAttribute("aria-expanded") === "true";
        const newExpandedState = !currentlyExpanded;
        toggleButton.setAttribute("aria-expanded", String(newExpandedState));
        const currentLogoElements = document.querySelectorAll('[data-classifier-logo="true"]');
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
    copyShareableLink() {
      ShareLink.copyShareableLink();
    }
  }
  function showInitialLoadingIndicator() {
    const indicator = document.getElementById("loading-indicator");
    if (indicator) {
      indicator.classList.add("htmx-request");
    }
  }
  document.addEventListener("DOMContentLoaded", () => {
    new ClassifierPage;
  });
  if (document.readyState === "complete" || document.readyState === "interactive") {
    new ClassifierPage;
  }
  if (document.readyState === "complete" || document.readyState === "interactive") {
    showInitialLoadingIndicator();
  }
  window.showInitialLoadingIndicator = showInitialLoadingIndicator;
})();
