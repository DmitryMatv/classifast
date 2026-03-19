import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("storefront.ts", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("redirects to Polar checkout with slug and absolute return url", async () => {
    document.body.innerHTML = `
      <button
        type="button"
        data-mapping-buy-button
        data-mapping-slug="unspsc-to-cpv-mapping"
        data-return-url="/mapping/unspsc-to-cpv-mapping/"
      >
        Buy full file
      </button>
    `;

    window.__storefrontNavigate = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ url: "https://polar.example/checkout" }),
      })),
    );

    await import("./storefront");
    const button = document.querySelector(
      "[data-mapping-buy-button]",
    ) as HTMLButtonElement;

    button.click();
    await Promise.resolve();
    await Promise.resolve();

    expect(fetch).toHaveBeenCalledWith("/api/create-mapping-checkout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slug: "unspsc-to-cpv-mapping",
        return_url: `${window.location.origin}/mapping/unspsc-to-cpv-mapping/`,
      }),
    });
    expect(window.__storefrontNavigate).toHaveBeenCalledWith(
      "https://polar.example/checkout",
    );
  });

  it("shows an error state and restores the button after a failed checkout", async () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <button
        type="button"
        data-mapping-buy-button
        data-mapping-slug="cpv-to-unspsc-mapping"
      >
        Buy with Polar
      </button>
    `;

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        json: async () => ({}),
      })),
    );

    await import("./storefront");
    const button = document.querySelector(
      "[data-mapping-buy-button]",
    ) as HTMLButtonElement;

    button.click();
    await Promise.resolve();
    await Promise.resolve();

    expect(button.textContent?.trim()).toBe("Error - Try again");

    vi.advanceTimersByTime(3000);

    expect(button.textContent?.trim()).toBe("Buy with Polar");
  });

  it("shows success banner and removes checkout params from the URL", async () => {
    const replaceStateSpy = vi.spyOn(window.history, "replaceState");
    window.history.replaceState(
      {},
      "",
      "/mapping/unspsc-to-cpv-mapping/?checkout=success&foo=bar",
    );
    document.body.innerHTML = `
      <div data-storefront-success class="hidden">Success banner</div>
    `;

    await import("./storefront");

    const banner = document.querySelector(
      "[data-storefront-success]",
    ) as HTMLElement;

    expect(banner.classList.contains("hidden")).toBe(false);
    expect(window.location.search).toBe("?foo=bar");
    expect(replaceStateSpy).toHaveBeenLastCalledWith(
      {},
      "",
      "/mapping/unspsc-to-cpv-mapping/?foo=bar",
    );
  });
});
