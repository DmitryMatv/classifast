import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("paywall.ts", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    document.body.innerHTML = `
      <div id="results-container"></div>
      <div id="paywall-warning"></div>
      <div id="paywall-buttons"></div>
      <form hx-get="/NAICS/fragment"></form>
      <button id="retry-button">Retry</button>
      <button id="signin-button">Sign In</button>
    `;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("prevents duplicate initialization", async () => {
    const { initPaywall } = await import("./paywall");
    const form = document.querySelector("form") as HTMLFormElement;
    const requestSubmitSpy = vi.mocked(form.requestSubmit);

    vi.runAllTimers();
    initPaywall();
    initPaywall();

    const retryButton = document.getElementById("retry-button") as HTMLElement;
    retryButton.click();

    expect(requestSubmitSpy).toHaveBeenCalledTimes(1);
  });

  it("retries classification when the retry button is clicked", async () => {
    const { initPaywall } = await import("./paywall");
    const form = document.querySelector("form") as HTMLFormElement;
    const requestSubmitSpy = vi.mocked(form.requestSubmit);

    vi.runAllTimers();
    initPaywall();
    (document.getElementById("retry-button") as HTMLElement).click();

    expect(requestSubmitSpy).toHaveBeenCalled();
  });

  it("opens the Clerk sign-in modal when the sign-in button is clicked", async () => {
    const { initPaywall } = await import("./paywall");

    vi.runAllTimers();
    initPaywall();
    (document.getElementById("signin-button") as HTMLElement).click();

    expect(window.Clerk?.openSignIn).toHaveBeenCalled();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("opens sign-in when the upgrade button is clicked while signed out", async () => {
    document.body.innerHTML +=
      '<button id="upgrade-button">Upgrade to Pro</button>';
    const { initPaywall } = await import("./paywall");

    vi.runAllTimers();
    initPaywall();
    (document.getElementById("upgrade-button") as HTMLButtonElement).click();

    expect(window.Clerk?.openSignIn).toHaveBeenCalled();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("reinitializes after HTMX swaps the results container", async () => {
    const { initPaywall } = await import("./paywall");
    const form = document.querySelector("form") as HTMLFormElement;
    const requestSubmitSpy = vi.mocked(form.requestSubmit);

    vi.runAllTimers();
    initPaywall();
    window.__paywallInitialized = true;

    document.body.dispatchEvent(
      new CustomEvent("htmx:after:swap", {
        detail: {
          ctx: { target: document.getElementById("results-container") },
        },
      } as CustomEventInit),
    );
    vi.runAllTimers();

    (document.getElementById("retry-button") as HTMLElement).click();

    expect(window.__paywallInitialized).toBe(true);
    expect(requestSubmitSpy).toHaveBeenCalled();
  });

  it("shows an error state when checkout starts without a Clerk session", async () => {
    document.body.innerHTML +=
      '<button id="upgrade-button">Upgrade to Pro</button>';
    window.Clerk!.user = { id: "user_123" };
    delete window.Clerk!.session;
    const { initPaywall } = await import("./paywall");

    vi.runAllTimers();
    initPaywall();
    (document.getElementById("upgrade-button") as HTMLButtonElement).click();

    expect(
      (document.getElementById("upgrade-button") as HTMLButtonElement)
        .innerHTML,
    ).toContain("Error - Try again");
  });

  it("redirects to checkout on a successful upgrade response", async () => {
    document.body.innerHTML +=
      '<button id="upgrade-button">Upgrade to Pro</button>';
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" };
      window.Clerk.session = {
        getToken: vi.fn(async () => "token-123"),
      };
    }
    window.__paywallNavigate = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ url: "https://billing.example/checkout" }),
      })),
    );
    const { initPaywall } = await import("./paywall");

    vi.runAllTimers();
    initPaywall();
    (document.getElementById("upgrade-button") as HTMLButtonElement).click();

    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    const [requestUrl, requestInit] =
      vi.mocked(globalThis.fetch).mock.calls[0] ?? [];
    expect(requestUrl).toBe("/api/create-checkout");
    expect(requestInit?.body).toBeDefined();
    expect(JSON.parse(String(requestInit?.body))).toEqual({
      return_url: "http://localhost:3000/",
    });
    await vi.waitFor(() =>
      expect(window.__paywallNavigate).toHaveBeenCalledWith(
        "https://billing.example/checkout",
      ),
    );
  });

  it("restores the upgrade button after a failed checkout request", async () => {
    document.body.innerHTML +=
      '<button id="upgrade-button">Upgrade to Pro</button>';
    if (window.Clerk) {
      window.Clerk.user = { id: "user_123" };
      window.Clerk.session = {
        getToken: vi.fn(async () => "token-123"),
      };
    }
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        json: async () => ({}),
      })),
    );
    const { initPaywall } = await import("./paywall");

    vi.runAllTimers();
    initPaywall();
    const button = document.getElementById(
      "upgrade-button",
    ) as HTMLButtonElement;
    button.click();
    await Promise.resolve();
    await Promise.resolve();

    expect(button.innerHTML).toContain("Error - Try again");

    vi.advanceTimersByTime(3000);
    expect(button.innerHTML).toContain("Upgrade to Pro");
    expect(button.disabled).toBe(false);
  });
});
