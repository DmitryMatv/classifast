import "./types/globals";

const restoreTimeouts = new WeakMap<
  HTMLButtonElement,
  Parameters<typeof clearTimeout>[0]
>();

function redirectToStorefrontUrl(url: string): void {
  if (window.__storefrontNavigate) {
    window.__storefrontNavigate(url);
    return;
  }

  window.location.assign(url);
}

function restoreButton(
  button: HTMLButtonElement,
  originalHtml: string,
  originalDisabled: boolean,
): void {
  button.innerHTML = originalHtml;
  button.disabled = originalDisabled;
}

async function startCheckout(button: HTMLButtonElement): Promise<void> {
  const slug = button.dataset["mappingSlug"];
  if (!slug) return;

  const existingRestoreTimeout = restoreTimeouts.get(button);
  if (existingRestoreTimeout) {
    clearTimeout(existingRestoreTimeout);
    restoreTimeouts.delete(button);
  }

  const originalHtml = button.innerHTML;
  const originalDisabled = button.disabled;
  const returnUrl = new URL(
    button.dataset["returnUrl"] || window.location.href,
    window.location.origin,
  ).toString();

  button.disabled = true;
  button.innerHTML = "Preparing...";

  try {
    const response = await fetch("/api/create-mapping-checkout", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        slug,
        return_url: returnUrl,
      }),
    });

    if (!response.ok) {
      throw new Error("Checkout creation failed");
    }

    const data = (await response.json()) as { url?: string };
    if (!data.url) {
      throw new Error("No checkout URL returned");
    }

    redirectToStorefrontUrl(data.url);
  } catch (error: unknown) {
    console.error("Mapping checkout failed:", error);
    button.innerHTML = "Error - Try again";
    button.disabled = false;

    const restoreTimeout = setTimeout(() => {
      restoreButton(button, originalHtml, originalDisabled);
      restoreTimeouts.delete(button);
    }, 3000);
    restoreTimeouts.set(button, restoreTimeout);
  }
}

function initializeBuyButtons(): void {
  const buttons = document.querySelectorAll<HTMLButtonElement>(
    "[data-mapping-buy-button]",
  );

  buttons.forEach((button) => {
    const replacement = button.cloneNode(true) as HTMLButtonElement;
    button.parentNode?.replaceChild(replacement, button);
    replacement.addEventListener("click", async (event) => {
      event.preventDefault();
      await startCheckout(replacement);
    });
  });
}

function showCheckoutSuccessMessage(): void {
  const successBanners = document.querySelectorAll<HTMLElement>(
    "[data-storefront-success]",
  );

  if (!successBanners.length) return;

  successBanners.forEach((banner) => {
    banner.classList.remove("hidden");
  });
}

function cleanupCheckoutParams(): void {
  const url = new URL(window.location.href);
  const params = url.searchParams;
  const isCheckoutSuccess = params.get("checkout") === "success";

  if (!isCheckoutSuccess) return;

  showCheckoutSuccessMessage();

  params.delete("checkout");
  params.delete("checkout_token");
  params.delete("customer_session_token");

  const cleanUrl = `${url.pathname}${params.toString() ? `?${params.toString()}` : ""}${url.hash}`;
  window.history.replaceState({}, "", cleanUrl);
}

export function initStorefront(): void {
  if (window.__storefrontInitialized) return;
  window.__storefrontInitialized = true;

  cleanupCheckoutParams();
  initializeBuyButtons();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    initStorefront();
  });
} else {
  initStorefront();
}
