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
    delete document.body.dataset["commonInitialized"];
    window.__authReady = false;
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
    await advanceTimersAndFlushAsync(4000);

    expect(window.__authReady).toBe(true);
    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("Sign In");
    expect(document.body.textContent).toContain("Sign Up");
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
    await advanceTimersAndFlushAsync(4000);

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

  it("preserves the hash when cleaning checkout params after successful auth bootstrap", async () => {
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
      "/NAICS/?checkout=success&foo=bar#results",
    );

    await import("./common");
    await flushAsyncWork();

    expect(window.location.pathname).toBe("/NAICS/");
    expect(window.location.search).toBe("?foo=bar");
    expect(window.location.hash).toBe("#results");
  });
});
