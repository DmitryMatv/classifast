import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ClerkHelpers } from "./clerk-helpers";

async function flushAsyncWork(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

async function advanceTimersAndFlushAsync(ms: number): Promise<void> {
  vi.advanceTimersByTime(ms);
  await flushAsyncWork();
}

function createJwtWithExpiration(exp: number): string {
  const payload = btoa(JSON.stringify({ exp }))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  return `header.${payload}.signature`;
}

describe("common.ts", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    const freshBody = document.body.cloneNode(false) as HTMLBodyElement;
    document.body.replaceWith(freshBody);
    document.body.innerHTML = "";
    document.body.removeAttribute("data-common-initialized");
    document.body.removeAttribute("data-auth-ui");
    delete document.body.dataset["commonInitialized"];
    delete document.body.dataset["authUi"];
    window.__authReady = false;
    delete window.__clerkScriptFailed;
    delete window.copyOriginalId;
    window.__internal_ClerkUICtor = {};
    window.self = window;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("uses clipboard copy and shows feedback on the share button", async () => {
    document.body.innerHTML = '<button id="share-button">Share</button>';
    const writeText = vi.mocked(navigator.clipboard.writeText);
    const { ShareLink } = await import("./common");

    await ShareLink.copyShareableLink();

    expect(writeText).toHaveBeenCalledWith("http://localhost:3000/");
    expect(document.getElementById("share-button")?.innerHTML).toBe("Copied!");

    vi.advanceTimersByTime(2000);

    expect(document.getElementById("share-button")?.innerHTML).toBe("Share");
  });

  it("falls back to execCommand when clipboard write fails", async () => {
    document.body.innerHTML = '<button id="share-button">Share</button>';
    vi.mocked(navigator.clipboard.writeText).mockRejectedValueOnce(
      new Error("copy failed"),
    );
    const execCommand = vi.mocked(document.execCommand);
    const { ShareLink } = await import("./common");

    await ShareLink.copyShareableLink();

    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(document.getElementById("share-button")?.innerHTML).toBe("Copied!");
  });

  it("toggles mobile menu and closes on Escape and outside click", async () => {
    document.body.innerHTML = `
      <button id="mobile-menu-button" class="hamburger" aria-expanded="false">
        <span></span>
        <span></span>
        <span></span>
      </button>
      <div id="mobile-menu"><a href="/x">Link</a></div>
    `;
    await import("./common");
    const button = document.getElementById("mobile-menu-button") as HTMLElement;
    const menu = document.getElementById("mobile-menu") as HTMLElement;

    button.click();
    expect(menu.classList.contains("active")).toBe(true);
    expect(button.getAttribute("aria-expanded")).toBe("true");
    expect(button.getAttribute("aria-controls")).toBe("mobile-menu");

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(menu.classList.contains("active")).toBe(false);

    button.click();
    document.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(menu.classList.contains("active")).toBe(false);
  });

  it("does not bind the mobile menu twice when initCommon runs again", async () => {
    document.body.innerHTML = `
      <button id="mobile-menu-button" class="hamburger" aria-expanded="false">
        <span></span>
        <span></span>
        <span></span>
      </button>
      <div id="mobile-menu"><a href="/x">Link</a></div>
    `;
    const { initCommon } = await import("./common");
    const button = document.getElementById("mobile-menu-button") as HTMLElement;
    const menu = document.getElementById("mobile-menu") as HTMLElement;

    initCommon();
    button.click();

    expect(menu.classList.contains("active")).toBe(true);
    expect(button.getAttribute("aria-expanded")).toBe("true");
  });

  it("submits textarea form on Enter but not Shift+Enter", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form>
        <textarea id="product_description_area"></textarea>
        <button type="submit">Submit</button>
      </form>
    `;
    await import("./common");
    const submitButton = document.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement;
    const clickSpy = vi.fn();
    submitButton.click = clickSpy;

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;

    textarea.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(clickSpy).toHaveBeenCalledTimes(1);

    textarea.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Enter",
        shiftKey: true,
        bubbles: true,
      }),
    );
    expect(clickSpy).toHaveBeenCalledTimes(1);
  });

  it("keeps textarea focus and selection when Enter submits the form", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form>
        <textarea id="product_description_area">Industrial pump</textarea>
        <button type="submit">Submit</button>
      </form>
    `;
    const form = document.querySelector("form") as HTMLFormElement;
    form.addEventListener("submit", (event) => event.preventDefault());

    await import("./common");

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.focus();
    textarea.setSelectionRange(10, 10);

    textarea.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );

    expect(document.activeElement).toBe(textarea);
    expect(textarea.selectionStart).toBe(10);
    expect(textarea.selectionEnd).toBe(10);
  });

  it("focuses a prefilled textarea with the cursor at the end", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <textarea id="product_description_area">Industrial pump</textarea>
    `;

    await import("./common");

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;

    expect(document.activeElement).toBe(textarea);
    expect(textarea.selectionStart).toBe(textarea.value.length);
    expect(textarea.selectionEnd).toBe(textarea.value.length);
  });

  it("does not steal existing focus while placing the cursor at the end", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <button id="existing-focus">Existing focus</button>
      <textarea id="product_description_area">Industrial pump</textarea>
    `;
    const existingFocus = document.getElementById(
      "existing-focus",
    ) as HTMLButtonElement;
    existingFocus.focus();

    await import("./common");

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;

    expect(document.activeElement).toBe(existingFocus);
    expect(textarea.selectionStart).toBe(textarea.value.length);
    expect(textarea.selectionEnd).toBe(textarea.value.length);
  });

  it("clears the default example text after the configured delay", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form data-default-example-prefill="true">
        <textarea id="product_description_area">Industrial pump</textarea>
      </form>
    `;

    await import("./common");

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;

    expect(textarea.value).toBe("Industrial pump");

    await advanceTimersAndFlushAsync(99);

    expect(textarea.value).toBe("Industrial pump");

    await advanceTimersAndFlushAsync(1);

    expect(textarea.value).toBe("");
    expect(textarea.defaultValue).toBe("");
    expect(textarea.textContent).toBe("");
  });

  it("does not clear the default example text if the user edits it before the timeout", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form data-default-example-prefill="true">
        <textarea id="product_description_area">Industrial pump</textarea>
      </form>
    `;

    await import("./common");

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.value = "Industrial pump updated";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));

    await advanceTimersAndFlushAsync(300);

    expect(textarea.value).toBe("Industrial pump updated");
  });

  it("does not schedule auto-clear for non-default prefilled text", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form data-default-example-prefill="false">
        <textarea id="product_description_area">helicopter taxi</textarea>
      </form>
    `;

    await import("./common");

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;

    await advanceTimersAndFlushAsync(1000);

    expect(textarea.value).toBe("helicopter taxi");
  });

  it("registers copyOriginalId and shows a tooltip when copying", async () => {
    document.body.innerHTML = '<button id="copy-button">Copy</button>';
    await import("./common");
    const button = document.getElementById("copy-button") as HTMLButtonElement;

    expect(window.copyOriginalId).toBeTypeOf("function");

    window.copyOriginalId?.("8471", button);
    await Promise.resolve();

    expect(vi.mocked(navigator.clipboard.writeText)).toHaveBeenCalledWith(
      "8471",
    );
    expect(document.body.textContent).toContain("Copied!");
  });

  it("falls back to execCommand when clipboard API is unavailable for result copy", async () => {
    document.body.innerHTML = '<button id="copy-button">Copy</button>';
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: undefined,
    });
    const execCommand = vi.mocked(document.execCommand);
    await import("./common");
    const button = document.getElementById("copy-button") as HTMLButtonElement;
    window.copyOriginalId?.("8471", button);

    expect(execCommand).toHaveBeenCalledWith("copy");
  });

  it("falls back to execCommand when clipboard write rejects for result copy", async () => {
    document.body.innerHTML = '<button id="copy-button">Copy</button>';
    vi.mocked(navigator.clipboard.writeText).mockRejectedValueOnce(
      new Error("clipboard denied"),
    );
    const execCommand = vi.mocked(document.execCommand);
    await import("./common");
    const button = document.getElementById("copy-button") as HTMLButtonElement;

    window.copyOriginalId?.("8471", button);
    await flushAsyncWork();

    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(document.body.textContent).toContain("Copied!");
  });

  it("shows copy failed when clipboard write rejects and fallback copy fails", async () => {
    document.body.innerHTML = '<button id="copy-button">Copy</button>';
    vi.mocked(navigator.clipboard.writeText).mockRejectedValueOnce(
      new Error("clipboard denied"),
    );
    vi.mocked(document.execCommand).mockReturnValueOnce(false);
    await import("./common");
    const button = document.getElementById("copy-button") as HTMLButtonElement;

    window.copyOriginalId?.("8471", button);
    await flushAsyncWork();

    expect(document.body.textContent).toContain("Copy failed");
  });

  it("submits forms through ClerkHelpers when present and returns false otherwise", () => {
    document.body.innerHTML = '<form hx-get="/"></form>';
    const form = document.querySelector("form") as HTMLFormElement;
    const requestSubmitSpy = vi.fn();
    form.requestSubmit = requestSubmitSpy;

    expect(ClerkHelpers.submitForm()).toBe(true);
    expect(requestSubmitSpy).toHaveBeenCalledTimes(1);

    document.body.innerHTML = "";
    expect(ClerkHelpers.submitForm()).toBe(false);
  });

  it("dispatches htmx:authReady after successful Clerk bootstrap", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    const authReadyListener = vi.fn();
    document.body.addEventListener("htmx:authReady", authReadyListener);
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" } as ClerkUser;
      window.Clerk.session = {
        getToken: vi.fn(async () => "token-123"),
      };
    }

    await import("./common");
    await flushAsyncWork();

    expect(window.Clerk?.load).toHaveBeenCalledWith({
      ui: {
        ClerkUI: window.__internal_ClerkUICtor,
      },
    });
    expect(window.Clerk?.session?.getToken).toHaveBeenCalled();
    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).not.toContain("Sign In");
    expect(window.Clerk?.openGoogleOneTap).not.toHaveBeenCalled();
  });

  it("opens Google One Tap with FedCM enabled for anonymous users", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;

    await import("./common");
    await flushAsyncWork();

    expect(window.Clerk?.openGoogleOneTap).toHaveBeenCalledTimes(1);
    expect(window.Clerk?.openGoogleOneTap).toHaveBeenCalledWith({
      cancelOnTapOutside: false,
      itpSupport: true,
      fedCmSupport: true,
    });
  });

  it("does not open Google One Tap when the user is already signed in", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" } as ClerkUser;
      window.Clerk.session = {
        getToken: vi.fn(async () => "token-123"),
      };
    }

    await import("./common");
    await flushAsyncWork();

    expect(window.Clerk?.openGoogleOneTap).not.toHaveBeenCalled();
  });

  it("opens Google One Tap only once across anonymous auth UI rerenders", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;

    let listener: (() => Promise<void>) | undefined;
    if (window.Clerk) {
      window.Clerk.addListener = vi.fn((callback) => {
        listener = callback as () => Promise<void>;
        return vi.fn();
      });
    }

    await import("./common");
    await flushAsyncWork();

    expect(window.Clerk?.openGoogleOneTap).toHaveBeenCalledTimes(1);
    expect(listener).toBeTypeOf("function");

    await listener?.();
    await flushAsyncWork();
    await listener?.();
    await flushAsyncWork();

    expect(window.Clerk?.openGoogleOneTap).toHaveBeenCalledTimes(1);
  });

  it("skips Google One Tap in an embedded browsing context", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    // Embedded context is detected via `window.self !== window`
    window.self = { embedded: true } as unknown as Window & typeof globalThis;

    await import("./common");
    await flushAsyncWork();

    expect(window.Clerk?.openGoogleOneTap).not.toHaveBeenCalled();
  });

  it("mounts Clerk user buttons without the extra trigger ring or background chrome", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" } as ClerkUser;
      window.Clerk.session = {
        getToken: vi.fn(async () => "token-123"),
      };
    }

    await import("./common");
    await flushAsyncWork();

    expect(window.Clerk?.mountUserButton).toHaveBeenCalledTimes(2);

    const desktopCall = vi.mocked(window.Clerk!.mountUserButton).mock.calls[0];
    const desktopOptions = desktopCall?.[1];
    const desktopTriggerClasses =
      desktopOptions?.appearance?.elements?.userButtonTrigger ?? "";

    expect(desktopTriggerClasses).toContain("focus:outline-none");
    expect(desktopTriggerClasses).toContain("focus-visible:ring-0");
    expect(desktopTriggerClasses).not.toContain("focus-visible:ring-2");
    expect(desktopTriggerClasses).not.toContain("ring-offset");
  });

  it("dispatches htmx:authReady when Clerk falls back", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    delete window.Clerk;
    const authReadyListener = vi.fn();
    document.body.addEventListener("htmx:authReady", authReadyListener);

    await import("./common");
    await flushAsyncWork();

    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("Sign In");
    expect(document.body.textContent).toContain("Sign Up");
  });

  it("falls back when Clerk.load hangs and still signals auth ready once", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    if (window.Clerk) {
      window.Clerk.load = vi.fn(
        () => new Promise<void>(() => undefined),
      ) as ClerkInstance["load"];
    }
    const authReadyListener = vi.fn();
    document.body.addEventListener("htmx:authReady", authReadyListener);

    await import("./common");
    await advanceTimersAndFlushAsync(10000);

    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("Sign In");
    expect(document.body.textContent).toContain("Sign Up");
  });

  it("falls back when the Clerk UI bundle never becomes available", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    delete window.__internal_ClerkUICtor;

    const authReadyListener = vi.fn();
    document.body.addEventListener("htmx:authReady", authReadyListener);

    await import("./common");
    await advanceTimersAndFlushAsync(10000);

    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("Sign In");
    expect(document.body.textContent).toContain("Sign Up");
  });

  it("starts Clerk when window.Clerk appears after a missed script load event", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    delete window.Clerk;

    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@clerk/clerk-js";
    document.head.appendChild(script);

    const commonModule = import("./common");
    await flushAsyncWork();
    await advanceTimersAndFlushAsync(5000);

    window.Clerk = {
      load: vi.fn(async () => {}),
      addListener: vi.fn(() => vi.fn()),
      mountUserButton: vi.fn(),
      openSignIn: vi.fn(),
      openSignUp: vi.fn(),
      openGoogleOneTap: vi.fn(),
    } as ClerkInstance;
    if (window.Clerk) {
      window.Clerk.session = {
        getToken: vi.fn(async () => "token-123"),
      };
    }
    await advanceTimersAndFlushAsync(100);
    await commonModule;

    expect(window.Clerk?.load).toHaveBeenCalledTimes(1);
    expect(consoleErrorSpy).not.toHaveBeenCalledWith(
      "Timed out waiting for Clerk script readiness",
    );
  });

  it("falls back immediately when Clerk script emits error before timeout", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    delete window.Clerk;

    const authReadyListener = vi.fn();
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    document.body.addEventListener("htmx:authReady", authReadyListener);
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@clerk/clerk-js";
    document.head.appendChild(script);

    const commonModule = await import("./common");
    commonModule.initCommon();
    await advanceTimersAndFlushAsync(50);

    script.dispatchEvent(new Event("error"));
    vi.runAllTimers();
    await flushAsyncWork();

    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("Sign In");
    expect(document.body.textContent).toContain("Sign Up");
    expect(consoleErrorSpy).toHaveBeenCalledWith("Clerk script failed to load");
    expect(consoleErrorSpy).not.toHaveBeenCalledWith(
      "Timed out waiting for Clerk script readiness",
    );
    expect(window.__clerkScriptFailed).toBe(true);
  });

  it("falls back immediately when Clerk script failure is already known before init", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    delete window.Clerk;
    window.__clerkScriptFailed = true;
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@clerk/clerk-js";
    document.head.appendChild(script);

    const authReadyListener = vi.fn();
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    document.body.addEventListener("htmx:authReady", authReadyListener);

    await import("./common");
    await flushAsyncWork();

    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("Sign In");
    expect(document.body.textContent).toContain("Sign Up");
    expect(consoleErrorSpy).toHaveBeenCalledWith("Clerk script failed to load");
  });

  it("starts Clerk as soon as window.Clerk appears without waiting for the full timeout", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    delete window.Clerk;

    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@clerk/clerk-js";
    document.head.appendChild(script);

    const commonModule = import("./common");
    await flushAsyncWork();
    await advanceTimersAndFlushAsync(500);

    window.Clerk = {
      load: vi.fn(async () => {}),
      addListener: vi.fn(() => vi.fn()),
      mountUserButton: vi.fn(),
      openSignIn: vi.fn(),
      openSignUp: vi.fn(),
      openGoogleOneTap: vi.fn(),
    } as ClerkInstance;
    if (window.Clerk) {
      window.Clerk.session = {
        getToken: vi.fn(async () => "token-123"),
      };
    }

    expect(window.Clerk?.load).not.toHaveBeenCalled();
    await advanceTimersAndFlushAsync(100);
    await commonModule;

    expect(window.Clerk?.load).toHaveBeenCalledTimes(1);
    expect(consoleErrorSpy).not.toHaveBeenCalledWith(
      "Timed out waiting for Clerk script readiness",
    );
  });

  it("falls back when initial token refresh hangs and still signals auth ready once", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" } as ClerkUser;
      window.Clerk.session = {
        getToken: vi.fn(() => new Promise<string | null>(() => undefined)),
      };
    }
    const authReadyListener = vi.fn();
    document.body.addEventListener("htmx:authReady", authReadyListener);

    await import("./common");
    await advanceTimersAndFlushAsync(10000);

    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("Sign In");
    expect(document.body.textContent).toContain("Sign Up");
  });

  it("renders semantic auth buttons that call Clerk helpers", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    const { ClerkHelpers: ImportedClerkHelpers } =
      await import("./clerk-helpers");
    const openSignInSpy = vi.spyOn(ImportedClerkHelpers, "openSignIn");
    const openSignUpSpy = vi.spyOn(ImportedClerkHelpers, "openSignUp");

    await import("./common");
    await flushAsyncWork();

    const signInButton = document.getElementById(
      "clerk-sign-in-button-desktop",
    ) as HTMLButtonElement;
    const signUpButton = document.getElementById(
      "clerk-sign-up-button-desktop",
    ) as HTMLButtonElement;

    expect(signInButton.tagName).toBe("BUTTON");
    expect(signInButton.type).toBe("button");
    expect(signUpButton.tagName).toBe("BUTTON");
    expect(signUpButton.type).toBe("button");

    signInButton.click();
    signUpButton.click();

    expect(openSignInSpy).toHaveBeenCalledTimes(1);
    expect(openSignUpSpy).toHaveBeenCalledTimes(1);
  });

  it("preserves checkout=success, strips sensitive checkout tokens, and keeps the hash after successful auth bootstrap", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
    `;
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" } as ClerkUser;
      window.Clerk.session = {
        getToken: vi.fn(async () => "token-123"),
      };
    }
    window.history.replaceState(
      {},
      "",
      "/NAICS/?checkout=success&checkout_token=checkout-secret&customer_session_token=customer-secret&foo=bar#results",
    );

    await import("./common");
    await flushAsyncWork();

    expect(window.location.pathname).toBe("/NAICS/");
    expect(window.location.search).toBe("?checkout=success&foo=bar");
    expect(window.location.hash).toBe("#results");
  });

  it("caches a refreshed Clerk session token", async () => {
    document.body.dataset["authUi"] = "disabled";
    const token = createJwtWithExpiration(Math.floor(Date.now() / 1000) + 60);
    if (window.Clerk) {
      window.Clerk.session = {
        getToken: vi.fn(async () => token),
      };
    }
    const { ClerkAuth } = await import("./common");

    await expect(ClerkAuth.refreshAuthToken()).resolves.toBe(token);

    expect(ClerkAuth.getCachedAuthToken()).toBe(token);
    expect(window.Clerk?.session?.getToken).toHaveBeenCalledWith({
      expirationBufferSeconds: 15,
    });
  });

  it("clears the cached Clerk token when session token refresh returns empty", async () => {
    document.body.dataset["authUi"] = "disabled";
    const token = createJwtWithExpiration(Math.floor(Date.now() / 1000) + 60);
    if (window.Clerk) {
      window.Clerk.session = {
        getToken: vi
          .fn()
          .mockResolvedValueOnce(token)
          .mockResolvedValueOnce(null),
      };
    }
    const { ClerkAuth } = await import("./common");

    await ClerkAuth.refreshAuthToken();
    await expect(ClerkAuth.refreshAuthToken()).resolves.toBeNull();

    expect(ClerkAuth.getCachedAuthToken()).toBeNull();
  });

  it("recovers a missing Clerk session before refreshing the token", async () => {
    document.body.dataset["authUi"] = "disabled";
    const recoveredToken = createJwtWithExpiration(
      Math.floor(Date.now() / 1000) + 60,
    );
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" } as ClerkUser;
      window.Clerk.load = vi.fn(async () => {
        if (window.Clerk) {
          window.Clerk.session = {
            getToken: vi.fn(async () => recoveredToken),
          };
        }
      });
      delete window.Clerk.session;
    }
    const { ClerkAuth } = await import("./common");

    await expect(ClerkAuth.refreshAuthToken()).resolves.toBe(recoveredToken);

    expect(window.Clerk?.load).toHaveBeenCalledWith({
      ui: {
        ClerkUI: window.__internal_ClerkUICtor,
      },
    });
    expect(ClerkAuth.getCachedAuthToken()).toBe(recoveredToken);
  });

  it("returns a still-valid cached token when token refresh fails", async () => {
    document.body.dataset["authUi"] = "disabled";
    const validCachedToken = createJwtWithExpiration(
      Math.floor(Date.now() / 1000) + 60,
    );
    if (window.Clerk) {
      window.Clerk.session = {
        getToken: vi
          .fn()
          .mockResolvedValueOnce(validCachedToken)
          .mockRejectedValueOnce(new Error("refresh failed")),
      };
    }
    const { ClerkAuth } = await import("./common");

    await ClerkAuth.refreshAuthToken();
    await expect(ClerkAuth.refreshAuthToken()).resolves.toBe(validCachedToken);
  });

  it("blocks and replays HTMX requests after refreshing a missing auth token", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
      <button
        id="retry-source"
        hx-post="/NAICS/fragment"
        hx-target="#results-container"
        hx-swap="outerHTML"
      ></button>
    `;
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" } as ClerkUser;
      window.Clerk.session = {
        getToken: vi
          .fn()
          .mockResolvedValueOnce(null)
          .mockResolvedValueOnce("retry-token"),
      };
    }
    await import("./common");
    await flushAsyncWork();

    const source = document.getElementById("retry-source") as HTMLElement;
    const event = new CustomEvent("htmx:configRequest", {
      bubbles: true,
      cancelable: true,
      detail: {
        headers: {},
        xhr: {},
        elt: source,
        parameters: {},
      },
    });

    document.body.dispatchEvent(event);
    await flushAsyncWork();

    expect(event.defaultPrevented).toBe(true);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    expect(window.htmx?.ajax).toHaveBeenCalledWith("POST", "/NAICS/fragment", {
      source,
      target: "#results-container",
      swap: "outerHTML",
    });
  });

  it("dispatches authRefreshFailed and skips HTMX replay when retry token refresh fails", async () => {
    document.body.innerHTML = `
      <div id="desktop-auth-container"></div>
      <div id="mobile-auth-container"></div>
      <button id="retry-source" hx-get="/NAICS/fragment"></button>
    `;
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" } as ClerkUser;
      window.Clerk.session = {
        getToken: vi.fn(async () => null),
      };
    }
    const refreshFailedListener = vi.fn();
    document.body.addEventListener(
      "htmx:authRefreshFailed",
      refreshFailedListener,
    );
    await import("./common");
    await flushAsyncWork();

    const source = document.getElementById("retry-source") as HTMLElement;
    document.body.dispatchEvent(
      new CustomEvent("htmx:configRequest", {
        bubbles: true,
        cancelable: true,
        detail: {
          headers: {},
          xhr: {},
          elt: source,
          parameters: {},
        },
      }),
    );
    await flushAsyncWork();

    expect(refreshFailedListener).toHaveBeenCalledTimes(1);
    expect(window.htmx?.ajax).not.toHaveBeenCalled();
  });
});
