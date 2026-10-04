import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function form(): HTMLFormElement {
  const element = document.querySelector("form[hx-get]");
  if (!(element instanceof HTMLFormElement)) throw new Error("Missing form");
  return element;
}

function button(id: string): HTMLButtonElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLButtonElement)) throw new Error(`Missing ${id}`);
  return element;
}

function renderPaywall(): void {
  const target = document.getElementById("results-container");
  if (!target) throw new Error("Missing results target");
  target.innerHTML = `<div id="paywall-warning"><h3>Free trial limit reached</h3><p>Upgrade for unlimited searches.</p>
    <div id="paywall-buttons"><button id="signin-button">Sign In</button><button id="upgrade-button">Upgrade to Pro</button></div>
    <button id="retry-button">Try again</button></div>`;
}

type TestRequest = HtmxRequestContext & {
  request: HtmxRequestContext["request"] & {
    signal: AbortSignal;
    abort: () => void;
  };
};

function request(sourceEvent?: Event): TestRequest {
  const source = form();
  const target = document.getElementById("results-container");
  if (!target) throw new Error("Missing results target");
  const controller = new AbortController();
  return {
    sourceElement: source,
    ...(sourceEvent ? { sourceEvent } : {}),
    target,
    swap: "innerHTML",
    request: {
      action: "/NAICS/fragment",
      method: "GET",
      headers: { "HX-Request": "true" },
      body: new FormData(source),
      signal: controller.signal,
      abort: () => controller.abort(),
    },
  } satisfies HtmxRequestContext;
}

function emit(name: string, ctx: HtmxRequestContext): void {
  document.body.dispatchEvent(new CustomEvent(name, { detail: { ctx } }));
}

function complete(
  ctx: HtmxRequestContext,
  status: number,
  html?: string,
): void {
  ctx.response = { status, headers: new Headers() };
  emit("htmx:after:request", ctx);
  if (html !== undefined) ctx.target.innerHTML = html;
  emit("htmx:after:swap", ctx);
  emit("htmx:finally:request", ctx);
}

async function flush(): Promise<void> {
  for (let i = 0; i < 8; i++) await Promise.resolve();
}

async function initialize(): Promise<() => void> {
  const { initPaywall } = await import("./paywall");
  initPaywall();
  await vi.advanceTimersByTimeAsync(0);
  return initPaywall;
}

function notify(user: ClerkUser | undefined, session?: ClerkSession): void {
  const clerk = window.Clerk;
  if (!clerk) throw new Error("Missing Clerk");
  if (user) clerk.user = user;
  else delete clerk.user;
  if (session) clerk.session = session;
  else delete clerk.session;
  const payload = {
    ...(user ? { user } : {}),
    ...(session ? { session } : {}),
  };
  for (const [listener] of vi.mocked(clerk.addListener).mock.calls)
    listener(payload);
}

describe("paywall.ts", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    document.body.replaceWith(document.createElement("body"));
    document.body.innerHTML = `<form hx-get="/NAICS/fragment" hx-target="#results-container">
      <textarea name="product_description">coffee</textarea><input name="top_k" value="5"></form>
      <div id="results-container"></div>`;
    renderPaywall();
    window.history.replaceState({}, "", "/");
    sessionStorage.clear();
    delete window.__checkoutReturnUrl;
  });

  afterEach(() => {
    window.dispatchEvent(new Event("pagehide"));
    delete window.__checkoutReturnUrl;
    sessionStorage.clear();
    vi.useRealTimers();
  });

  it("binds each current button once across repeated initialization and swaps", async () => {
    const init = await initialize();
    init();
    init();
    button("retry-button").click();
    expect(form().requestSubmit).toHaveBeenCalledTimes(1);
    renderPaywall();
    emit("htmx:after:swap", request());
    await vi.advanceTimersByTimeAsync(0);
    button("retry-button").click();
    expect(form().requestSubmit).toHaveBeenCalledTimes(2);
  });

  it("continues anonymous Upgrade once after a usable session without retrying classification", async () => {
    window.__paywallNavigate = vi.fn();
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ url: "https://billing.example/checkout" })),
    );
    const init = await initialize();
    button("upgrade-button").click();
    expect(window.Clerk?.openSignIn).toHaveBeenCalledTimes(1);
    const user = { id: "user_123" };
    notify(user);
    await flush();
    expect(fetch).not.toHaveBeenCalled();
    expect(form().requestSubmit).not.toHaveBeenCalled();
    document.getElementById("results-container")?.replaceChildren();
    init();
    const session = { getToken: vi.fn(async () => "token-123") };
    notify(user, session);
    notify(user, session);
    await flush();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(form().requestSubmit).not.toHaveBeenCalled();
    expect(fetch).toHaveBeenCalledWith(
      "/api/create-checkout",
      expect.objectContaining({
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer token-123",
        },
        body: JSON.stringify({ return_url: "http://localhost:3000/" }),
      }),
    );
    expect(window.__paywallNavigate).toHaveBeenCalledWith(
      "https://billing.example/checkout",
    );
  });

  it("claims a saved Upgrade on auth readiness even without paywall controls", async () => {
    await initialize();
    button("upgrade-button").click();
    expect(sessionStorage.length).toBe(1);
    window.dispatchEvent(new Event("pagehide"));
    document.body.replaceWith(document.createElement("body"));
    document.body.innerHTML =
      '<form hx-get="/NAICS/fragment"></form><div id="results-container">Results</div>';
    delete window.__paywallScriptParsed;
    delete window.__initPaywall;
    delete window.__paywallClerkListenerRegistered;
    vi.resetModules();
    vi.mocked(window.Clerk!.addListener).mockClear();
    window.Clerk!.user = { id: "user_123" };
    window.Clerk!.session = { getToken: vi.fn(async () => "token-123") };
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ url: "https://billing.example/checkout" })),
    );
    window.__paywallNavigate = vi.fn();
    await initialize();
    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    document.body.dispatchEvent(new CustomEvent("htmx:authReady"));
    await flush();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(sessionStorage.length).toBe(0);
  });

  it("ordinary Sign In clears Upgrade and retries lookup on the later usable-session notification", async () => {
    await initialize();
    button("upgrade-button").click();
    button("signin-button").click();
    expect(sessionStorage.length).toBe(0);
    const user = { id: "user_123" };
    notify(user);
    expect(form().requestSubmit).not.toHaveBeenCalled();
    notify(user, { getToken: vi.fn(async () => "token") });
    notify(user, window.Clerk?.session);
    expect(form().requestSubmit).toHaveBeenCalledTimes(1);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("does not automatically recreate checkout after a failed POST and renders the current button", async () => {
    const user = { id: "user_123" };
    const session = { getToken: vi.fn(async () => "token") };
    notify(user, session);
    let finish: (response: Response) => void = () => {};
    vi.mocked(fetch).mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const init = await initialize();
    button("upgrade-button").click();
    button("upgrade-button").click();
    await flush();
    renderPaywall();
    init();
    expect(button("upgrade-button").disabled).toBe(true);
    finish(new Response("", { status: 503 }));
    await flush();
    expect(button("upgrade-button").textContent).toContain("Error - Try again");
    notify(user, session);
    await vi.advanceTimersByTimeAsync(3000);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(button("upgrade-button").disabled).toBe(false);
    expect(button("upgrade-button").textContent).toContain("Upgrade to Pro");
  });

  it("retains Upgrade while a user notification has no session", async () => {
    notify({ id: "user_123" });
    await initialize();
    button("upgrade-button").click();
    expect(fetch).not.toHaveBeenCalled();
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ url: "https://billing.example/checkout" })),
    );
    window.__paywallNavigate = vi.fn();
    notify({ id: "user_123" }, { getToken: vi.fn(async () => "token") });
    await flush();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("creates authenticated checkout with the current return URL", async () => {
    window.history.replaceState({}, "", "/NAICS/coffee?top_k=5");
    notify({ id: "user_123" }, { getToken: vi.fn(async () => "token-123") });
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ url: "https://billing.example/checkout" })),
    );
    window.__paywallNavigate = vi.fn();
    await initialize();
    button("upgrade-button").click();
    await flush();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith(
      "/api/create-checkout",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          return_url: "http://localhost:3000/NAICS/coffee?top_k=5",
        }),
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer token-123",
        },
      }),
    );
    expect(window.__paywallNavigate).toHaveBeenCalledWith(
      "https://billing.example/checkout",
    );
  });

  it("fails closed without a token and permits an explicit checkout retry", async () => {
    const token = vi.fn(async (): Promise<string | null> => null);
    notify({ id: "user_123" }, { getToken: token });
    await initialize();
    button("upgrade-button").click();
    await flush();
    expect(button("upgrade-button").textContent).toContain("Error - Try again");
    expect(button("upgrade-button").disabled).toBe(false);
    expect(fetch).not.toHaveBeenCalled();
    token.mockResolvedValue("token-123");
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ url: "https://billing.example/checkout" })),
    );
    window.__paywallNavigate = vi.fn();
    button("upgrade-button").click();
    await flush();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(window.__paywallNavigate).toHaveBeenCalledWith(
      "https://billing.example/checkout",
    );
  });

  it("clears Upgrade before a header Sign In can synchronously notify usable authentication", async () => {
    const headerSignIn = document.createElement("button");
    headerSignIn.id = "clerk-sign-in-button-desktop";
    headerSignIn.addEventListener("click", () =>
      notify({ id: "user_123" }, { getToken: vi.fn(async () => "token") }),
    );
    document.body.append(headerSignIn);
    await initialize();
    button("upgrade-button").click();
    headerSignIn.click();
    await flush();
    expect(fetch).not.toHaveBeenCalled();
    expect(form().requestSubmit).toHaveBeenCalledTimes(1);
    expect(sessionStorage.length).toBe(0);
  });

  it("times out token waiting without issuing a later checkout", async () => {
    let resolveToken: (token: string) => void = () => {};
    notify(
      { id: "user_123" },
      {
        getToken: vi.fn(
          () =>
            new Promise<string>((resolve) => {
              resolveToken = resolve;
            }),
        ),
      },
    );
    await initialize();
    button("upgrade-button").click();
    await vi.advanceTimersByTimeAsync(30000);
    expect(button("upgrade-button").textContent).toContain("Error - Try again");
    resolveToken("late-token");
    await flush();
    expect(fetch).not.toHaveBeenCalled();
    expect(sessionStorage.length).toBe(0);
  });

  it("discards a saved Upgrade when returning to a different route", async () => {
    const init = await initialize();
    button("upgrade-button").click();
    window.history.replaceState({}, "", "/NAICS/coffee");
    init();
    notify({ id: "user_123" }, { getToken: vi.fn(async () => "token") });
    await flush();
    expect(sessionStorage.length).toBe(0);
    expect(fetch).not.toHaveBeenCalled();
    expect(form().requestSubmit).toHaveBeenCalledTimes(1);
  });

  it("observes initial autoload and serializes pending activation retries without checkout", async () => {
    window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
    const init = await initialize();
    expect(button("upgrade-button").hidden).toBe(true);
    expect(document.getElementById("paywall-warning")?.textContent).toContain(
      "Payment activation is pending",
    );
    expect(window.htmx?.ajax).not.toHaveBeenCalled();
    const initial = request();
    emit("htmx:config:request", initial);
    initial.response = { status: 429, headers: new Headers() };
    emit("htmx:after:request", initial);
    await vi.advanceTimersByTimeAsync(2500);
    expect(window.htmx?.ajax).not.toHaveBeenCalled();
    renderPaywall();
    emit("htmx:after:swap", initial);
    expect(button("upgrade-button").hidden).toBe(true);
    emit("htmx:finally:request", initial);
    let retry = request();
    vi.mocked(window.htmx!.ajax).mockImplementation(
      async (_method, _url, options) => {
        retry = request(options.event);
        emit("htmx:config:request", retry);
        emit("htmx:before:request", retry);
      },
    );
    await vi.advanceTimersByTimeAsync(2000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    expect(window.htmx?.ajax).toHaveBeenCalledWith(
      "GET",
      "/NAICS/fragment",
      expect.objectContaining({
        source: form(),
        target: "#results-container",
        values: { push_url: "false" },
      }),
    );
    init();
    await vi.advanceTimersByTimeAsync(10000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    complete(retry, 200, '<p id="no-results">No results found.</p>');
    await vi.advanceTimersByTimeAsync(10000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    expect(fetch).not.toHaveBeenCalled();
    expect(sessionStorage.length).toBe(0);
    expect(document.getElementById("no-results")?.textContent).toBe(
      "No results found.",
    );
    expect(document.getElementById("paywall-warning")).toBeNull();
  });

  it("retries 503 and network failures within one deadline and exposes manual Try again", async () => {
    window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
    const init = await initialize();
    const initial = request();
    emit("htmx:config:request", initial);
    complete(initial, 503, "Service unavailable");
    vi.mocked(window.htmx!.ajax).mockRejectedValue(
      new Error("Network failure"),
    );
    await vi.advanceTimersByTimeAsync(2000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    init();
    await vi.advanceTimersByTimeAsync(58000);
    const count = vi.mocked(window.htmx!.ajax).mock.calls.length;
    expect(count).toBeGreaterThan(1);
    await vi.advanceTimersByTimeAsync(10000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(count);
    expect(document.getElementById("results-container")?.textContent).toContain(
      "Payment activation is still pending. Try again.",
    );
    expect(document.getElementById("upgrade-button")).toBeNull();
    expect(button("retry-button").disabled).toBe(false);
    button("retry-button").click();
    await flush();
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(count + 1);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("bounds a hung initial lookup and aborts recovery on edited forms", async () => {
    window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
    await initialize();
    const initial = request();
    emit("htmx:config:request", initial);
    await vi.advanceTimersByTimeAsync(60000);
    expect(initial.request.signal.aborted).toBe(true);
    expect(button("retry-button").disabled).toBe(false);
    let retry = request();
    vi.mocked(window.htmx!.ajax).mockImplementation(
      async (_method, _url, options) => {
        retry = request(options.event);
        emit("htmx:config:request", retry);
      },
    );
    button("retry-button").click();
    await flush();
    form()
      .querySelector("textarea")
      ?.dispatchEvent(new Event("input", { bubbles: true }));
    expect(retry.request.signal.aborted).toBe(true);
    await vi.advanceTimersByTimeAsync(60000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    expect(sessionStorage.length).toBe(0);
  });

  it.each([
    ["initial", "input"],
    ["initial", "change"],
    ["waiting", "input"],
    ["waiting", "change"],
    ["requesting", "input"],
    ["requesting", "change"],
  ])(
    "shows manual retry after cancelling %s recovery with %s",
    async (phase, eventName) => {
      window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
      const init = await initialize();
      const warning = document.getElementById("paywall-warning");
      const initial = request();
      if (phase !== "initial") {
        emit("htmx:config:request", initial);
        complete(initial, 429);
      }
      let retry = request();
      vi.mocked(window.htmx!.ajax).mockImplementation(
        async (_method, _url, options) => {
          retry = request(options.event);
          emit("htmx:config:request", retry);
        },
      );
      if (phase === "requesting") await vi.advanceTimersByTimeAsync(2000);
      expect(warning?.textContent).toContain("Checking automatically...");
      expect(button("retry-button").disabled).toBe(true);
      const input = form().querySelector("textarea");
      if (!input) throw new Error("Missing description input");
      input.value = "updated coffee";
      input.dispatchEvent(new Event(eventName, { bubbles: true }));
      expect(button("retry-button").disabled).toBe(false);
      expect(form().querySelector("textarea")).toBe(input);
      expect(input.value).toBe("updated coffee");
      expect(document.getElementById("paywall-warning")).toBe(warning);
      expect(warning?.querySelector("[role='status']")?.textContent).toBe(
        "Payment activation is still pending. Try again.",
      );
      expect(warning?.textContent).not.toContain("Checking automatically...");
      if (phase === "requesting") {
        expect(retry.request.signal.aborted).toBe(true);
        complete(retry, 429);
      }
      init();
      await vi.advanceTimersByTimeAsync(60000);
      expect(window.htmx?.ajax).toHaveBeenCalledTimes(
        phase === "requesting" ? 1 : 0,
      );
      expect(sessionStorage.length).toBe(0);
      button("retry-button").click();
      expect(form().requestSubmit).toHaveBeenCalledTimes(1);
    },
  );

  it("ignores unrelated swaps when observing the initial response", async () => {
    window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
    await initialize();
    const initial = request();
    emit("htmx:config:request", initial);
    complete(request(), 200, "Unrelated response");
    complete(initial, 429);
    await vi.advanceTimersByTimeAsync(2000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
  });

  it("retires recovery when the current form is replaced and ignores its late response", async () => {
    window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
    const init = await initialize();
    const initial = request();
    emit("htmx:config:request", initial);
    complete(initial, 429);
    let retry = request();
    vi.mocked(window.htmx!.ajax).mockImplementation(
      async (_method, _url, options) => {
        retry = request(options.event);
        emit("htmx:config:request", retry);
      },
    );
    await vi.advanceTimersByTimeAsync(2000);
    const replacement = document.createElement("form");
    replacement.setAttribute("hx-get", "/NAICS/fragment");
    form().replaceWith(replacement);
    init();
    expect(retry.request.signal.aborted).toBe(true);
    complete(retry, 200);
    await vi.advanceTimersByTimeAsync(60000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    expect(sessionStorage.length).toBe(0);
  });

  it("preserves the absolute recovery deadline across a page return", async () => {
    window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
    await initialize();
    const initial = request();
    emit("htmx:config:request", initial);
    complete(initial, 429);
    await vi.advanceTimersByTimeAsync(40000);
    window.dispatchEvent(new Event("pagehide"));
    window.dispatchEvent(new Event("pageshow"));
    await vi.advanceTimersByTimeAsync(20000);
    expect(button("retry-button").disabled).toBe(false);
    expect(document.getElementById("paywall-warning")?.textContent).toContain(
      "still pending",
    );
    const count = vi.mocked(window.htmx!.ajax).mock.calls.length;
    await vi.advanceTimersByTimeAsync(60000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(count);
  });

  it.each([
    [
      "/NAICS/industrial_pump/?top_k=10&checkout=success",
      "/NAICS/industrial_pump/",
    ],
    ["/NAICS/?product_description=pump&checkout=success", "/NAICS/pump/"],
    [
      "/NAICS/?product_description=pump&enhance_query=1&checkout=success",
      "/NAICS/pump/?enhance_query=1",
    ],
  ])(
    "keeps activation recovery when its initial response canonicalizes %s",
    async (returnPath, canonicalPath) => {
      window.history.replaceState({}, "", returnPath);
      window.__checkoutReturnUrl = window.location.href;
      const deadline = Date.now() + 60000;
      const init = await initialize();
      const initial = request();
      emit("htmx:config:request", initial);
      initial.response = {
        status: 429,
        headers: new Headers({ "HX-Push-Url": canonicalPath }),
      };
      emit("htmx:after:request", initial);
      window.history.pushState({}, "", canonicalPath);
      renderPaywall();
      emit("htmx:after:swap", initial);
      emit("htmx:finally:request", initial);
      expect(button("upgrade-button").hidden).toBe(true);
      expect(
        JSON.parse(
          sessionStorage.getItem("classifast:checkout-recovery") ?? "null",
        ),
      ).toEqual({
        returnUrl: window.location.href,
        deadline,
      });
      await vi.advanceTimersByTimeAsync(2000);
      expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
      window.history.replaceState({}, "", "/ETIM/other/");
      init();
      await vi.advanceTimersByTimeAsync(2000);
      expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
      expect(sessionStorage.length).toBe(0);
    },
  );

  it.each(["https://other.example/NAICS/pump/", "/NAICS/other/"])(
    "rejects an owned push URL that does not match the same-origin location, %s",
    async (pushUrl) => {
      window.history.replaceState(
        {},
        "",
        "/NAICS/?product_description=pump&checkout=success",
      );
      window.__checkoutReturnUrl = window.location.href;
      await initialize();
      const initial = request();
      emit("htmx:config:request", initial);
      initial.response = {
        status: 429,
        headers: new Headers({ "HX-Push-Url": pushUrl }),
      };
      emit("htmx:after:request", initial);
      window.history.pushState({}, "", "/NAICS/pump/");
      renderPaywall();
      emit("htmx:after:swap", initial);
      emit("htmx:finally:request", initial);
      await vi.advanceTimersByTimeAsync(2000);
      expect(window.htmx?.ajax).not.toHaveBeenCalled();
      expect(button("upgrade-button").hidden).toBe(false);
      expect(document.querySelector("#paywall-warning p")?.textContent).toBe(
        "Upgrade for unlimited searches.",
      );
      expect(sessionStorage.length).toBe(0);
    },
  );

  it("resumes one immediate serialized lookup on a persisted return to the retained form", async () => {
    window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
    const deadline = Date.now() + 60000;
    const init = await initialize();
    const initial = request();
    emit("htmx:config:request", initial);
    complete(initial, 429);
    window.dispatchEvent(
      new PageTransitionEvent("pagehide", { persisted: true }),
    );
    await vi.advanceTimersByTimeAsync(10000);
    let retry = request();
    vi.mocked(window.htmx!.ajax).mockImplementation(
      async (_method, _url, options) => {
        retry = request(options.event);
        emit("htmx:config:request", retry);
      },
    );
    window.dispatchEvent(
      new PageTransitionEvent("pageshow", { persisted: true }),
    );
    await flush();
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    expect(
      JSON.parse(
        sessionStorage.getItem("classifast:checkout-recovery") ?? "null",
      ),
    ).toMatchObject({ deadline });
    init();
    window.dispatchEvent(
      new PageTransitionEvent("pageshow", { persisted: true }),
    );
    await vi.advanceTimersByTimeAsync(2000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
    complete(retry, 429);
    await vi.advanceTimersByTimeAsync(2000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(2);
  });

  it("leaves a fresh pageshow to the classifier's initial autoload", async () => {
    window.__checkoutReturnUrl = "http://localhost:3000/?checkout=success";
    await initialize();
    window.dispatchEvent(
      new PageTransitionEvent("pageshow", { persisted: false }),
    );
    await flush();
    expect(window.htmx?.ajax).not.toHaveBeenCalled();
    const initial = request();
    emit("htmx:config:request", initial);
    complete(initial, 429);
    await vi.advanceTimersByTimeAsync(2000);
    expect(window.htmx?.ajax).toHaveBeenCalledTimes(1);
  });
});
