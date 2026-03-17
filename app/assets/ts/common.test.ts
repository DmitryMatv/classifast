import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ClerkHelpers } from "./clerk-helpers";

async function flushAsyncWork(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

describe("common.ts", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
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
      <button id="mobile-menu-button" aria-expanded="false"></button>
      <div id="mobile-menu"><a href="/x">Link</a></div>
      <div class="hamburger"></div>
    `;
    await import("./common");
    const button = document.getElementById("mobile-menu-button") as HTMLElement;
    const menu = document.getElementById("mobile-menu") as HTMLElement;

    button.click();
    expect(menu.classList.contains("active")).toBe(true);
    expect(button.getAttribute("aria-expanded")).toBe("true");

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(menu.classList.contains("active")).toBe(false);

    button.click();
    document.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(menu.classList.contains("active")).toBe(false);
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
    window.Clerk = {
      load: vi.fn(async () => {}),
      user: { id: "user_123" } as ClerkUser,
      session: {
        getToken: vi.fn(async () => "token-123"),
      },
    } as unknown as typeof window.Clerk;

    await import("./common");
    await flushAsyncWork();

    expect(window.Clerk?.load).toHaveBeenCalled();
    expect(window.Clerk?.session?.getToken).toHaveBeenCalled();
    expect(authReadyListener).toHaveBeenCalledTimes(1);
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

    expect(authReadyListener).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("Sign In");
    expect(document.body.textContent).toContain("Sign Up");
  });
});
