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

describe("common.ts", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    document.body.innerHTML = "";
    document.body.removeAttribute("data-common-initialized");
    document.body.removeAttribute("data-auth-ui");
    delete document.body.dataset["commonInitialized"];
    delete document.body.dataset["authUi"];
    window.__authReady = false;
    delete window.__clerkScriptFailed;
    delete window.copyOriginalId;
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

  it("moves the caret to the end of initial prefilled text when already focused on init", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form data-default-example-prefill="true">
        <textarea id="product_description_area" autofocus>Industrial pump</textarea>
      </form>
    `;

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    textarea.focus();

    await import("./common");

    const expectedPosition = textarea.value.length;

    expect(document.activeElement).toBe(textarea);
    expect(textarea.selectionStart).toBe(expectedPosition);
    expect(textarea.selectionEnd).toBe(expectedPosition);
  });

  it("moves the caret to the end of initial prefilled text when focus arrives after init", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form data-default-example-prefill="false">
        <textarea id="product_description_area" autofocus>helicopter taxi</textarea>
      </form>
    `;

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const setSelectionRangeSpy = vi.spyOn(textarea, "setSelectionRange");

    await import("./common");

    expect(setSelectionRangeSpy).not.toHaveBeenCalled();

    textarea.focus();

    const expectedPosition = textarea.value.length;

    expect(document.activeElement).toBe(textarea);
    expect(textarea.selectionStart).toBe(expectedPosition);
    expect(textarea.selectionEnd).toBe(expectedPosition);
    expect(setSelectionRangeSpy).toHaveBeenCalledWith(
      expectedPosition,
      expectedPosition,
    );
  });

  it("moves the caret to the end for non-default prefilled text", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form data-default-example-prefill="false">
        <textarea id="product_description_area" autofocus>industrial pump</textarea>
      </form>
    `;

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const setSelectionRangeSpy = vi.spyOn(textarea, "setSelectionRange");

    await import("./common");
    textarea.focus();

    const expectedPosition = textarea.value.length;

    expect(setSelectionRangeSpy).toHaveBeenCalledWith(
      expectedPosition,
      expectedPosition,
    );
    expect(textarea.selectionStart).toBe(expectedPosition);
    expect(textarea.selectionEnd).toBe(expectedPosition);
  });

  it("does not move the caret for an empty default example textarea", async () => {
    document.body.dataset["authUi"] = "disabled";
    document.body.innerHTML = `
      <form data-default-example-prefill="true">
        <textarea id="product_description_area" autofocus></textarea>
      </form>
    `;

    const textarea = document.getElementById(
      "product_description_area",
    ) as HTMLTextAreaElement;
    const setSelectionRangeSpy = vi.spyOn(textarea, "setSelectionRange");

    await import("./common");
    textarea.focus();

    expect(setSelectionRangeSpy).not.toHaveBeenCalled();
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

    await advanceTimersAndFlushAsync(999);

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

    await advanceTimersAndFlushAsync(1000);

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

    expect(window.Clerk?.load).toHaveBeenCalled();
    expect(window.Clerk?.session?.getToken).toHaveBeenCalled();
    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).not.toContain("Sign In");
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
});
