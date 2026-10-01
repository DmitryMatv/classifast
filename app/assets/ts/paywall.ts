import "./types/globals";
import { ClerkHelpers } from "./clerk-helpers";

export function initPaywall(): void {
  window.__initPaywall?.();
}

function redirectToPaywallUrl(url: string): void {
  if (window.__paywallNavigate) window.__paywallNavigate(url);
  else window.location.assign(url);
}

if (!window.__paywallScriptParsed) {
  window.__paywallScriptParsed = true;
  const UPGRADE_INTENT_KEY = "classifast:upgrade-intent";
  const RECOVERY_KEY = "classifast:checkout-recovery";
  const RECOVERY_DURATION_MS = 60000;
  const RECOVERY_INTERVAL_MS = 2000;

  type CheckoutOperation =
    | { kind: "idle" }
    | { kind: "error" }
    | { kind: "awaiting-signin"; returnUrl: string }
    | {
        kind: "creating";
        returnUrl: string;
        userId: string;
        controller: AbortController;
      }
    | { kind: "redirecting" };
  type RecoveryAttempt =
    | { kind: "initial" }
    | { kind: "starting"; event: Event; controller: AbortController }
    | {
        kind: "requesting";
        ctx: HtmxRequestContext;
        abort: () => void;
        recovered: boolean;
        initial: boolean;
      }
    | { kind: "waiting"; timer: number }
    | { kind: "delayed" };
  type Recovery = {
    returnUrl: string;
    deadline: number;
    form: HTMLFormElement;
    deadlineTimer: number;
    attempt: RecoveryAttempt;
  };

  class PaywallManager {
    private operation: CheckoutOperation = { kind: "idle" };
    private recovery: Recovery | null = null;
    private wasUsable = !!(window.Clerk?.user && window.Clerk.session);
    private readonly retryControllers = new WeakMap<Event, AbortController>();
    private readonly listeners = new AbortController();
    private unsubscribeClerk: (() => void) | null = null;
    private returnHint: string | undefined;
    private suspendedForm: HTMLFormElement | null = null;

    constructor() {
      this.returnHint = window.__checkoutReturnUrl;
      const savedIntent = this.readStorage(UPGRADE_INTENT_KEY);
      if (savedIntent && this.matchesRoute(savedIntent)) {
        this.operation = { kind: "awaiting-signin", returnUrl: savedIntent };
      } else this.writeStorage(UPGRADE_INTENT_KEY, null);
      const options = { signal: this.listeners.signal };
      document.body.addEventListener(
        "click",
        (event) => this.handleClick(event),
        { ...options, capture: true },
      );
      document.body.addEventListener(
        "input",
        (event) => this.handleEdit(event),
        options,
      );
      document.body.addEventListener(
        "change",
        (event) => this.handleEdit(event),
        options,
      );
      document.body.addEventListener(
        "submit",
        (event) => {
          if (
            event.target === this.recovery?.form &&
            this.recovery.attempt.kind !== "initial"
          ) {
            this.stopRecovery();
          }
        },
        { ...options, capture: true },
      );
      document.body.addEventListener(
        "htmx:authReady",
        () => {
          this.refresh();
          this.authChanged(window.Clerk ?? {});
        },
        options,
      );
      document.body.addEventListener(
        "clerk:loaded",
        () => this.refresh(),
        options,
      );
      document.body.addEventListener(
        "htmx:config:request",
        (event) => this.configureRecovery(event),
        options,
      );
      document.body.addEventListener(
        "htmx:after:swap",
        (event) => {
          const recovery = this.recovery;
          if (
            recovery?.attempt.kind === "requesting" &&
            recovery.attempt.ctx === event.detail.ctx
          ) {
            const pushUrl =
              event.detail.ctx.response?.headers.get("HX-Push-Url");
            if (
              recovery.attempt.initial &&
              pushUrl &&
              pushUrl !== "true" &&
              pushUrl !== "false" &&
              recovery.form === this.currentForm() &&
              event.detail.ctx.sourceElement === recovery.form &&
              !event.detail.ctx.request.signal?.aborted
            ) {
              try {
                const canonical = new URL(pushUrl, recovery.returnUrl);
                if (
                  canonical.origin === window.location.origin &&
                  this.matchesRoute(canonical.href)
                ) {
                  recovery.returnUrl = this.cleanUrl(canonical.href);
                  this.writeStorage(
                    RECOVERY_KEY,
                    JSON.stringify({
                      returnUrl: recovery.returnUrl,
                      deadline: recovery.deadline,
                    }),
                  );
                }
              } catch {}
            }
            recovery.attempt.recovered =
              !!event.detail.ctx.response &&
              event.detail.ctx.response.status >= 200 &&
              event.detail.ctx.response.status < 300 &&
              !event.detail.ctx.target.querySelector("#paywall-warning");
          }
          if (
            event.detail.ctx.target.id === "results-container" ||
            (event.detail.ctx.target === document.body &&
              event.detail.ctx.request.headers["HX-History-Restore-Request"] ===
                "true")
          ) {
            this.refresh();
          }
        },
        options,
      );
      document.body.addEventListener(
        "htmx:finally:request",
        (event) => {
          const recovery = this.recovery;
          if (
            recovery?.attempt.kind === "requesting" &&
            recovery.attempt.ctx === event.detail.ctx
          ) {
            if (recovery.attempt.recovered) this.stopRecovery();
            else this.scheduleRecovery(recovery);
          }
        },
        options,
      );
      document.body.addEventListener(
        "htmx:before:history:restore",
        () => this.suspend(),
        options,
      );
      window.addEventListener("popstate", () => this.suspend(), options);
      window.addEventListener(
        "pagehide",
        () => {
          this.suspendedForm = this.recovery?.form ?? null;
          this.suspend();
        },
        options,
      );
      window.addEventListener(
        "pageshow",
        (event) => {
          const form = this.suspendedForm;
          this.suspendedForm = null;
          this.refresh();
          if (
            event.persisted &&
            form &&
            this.recovery?.form === form &&
            this.recovery.attempt.kind === "initial"
          ) {
            void this.requestRecovery(this.recovery);
          }
        },
        options,
      );
      this.refresh();
    }

    refresh(): void {
      this.setupClerkListener();
      if (
        this.operation.kind === "awaiting-signin" &&
        !this.matchesRoute(this.operation.returnUrl)
      ) {
        this.clearUpgrade();
      }
      if (
        this.recovery &&
        (this.recovery.form !== this.currentForm() ||
          !this.matchesRoute(this.recovery.returnUrl))
      ) {
        this.stopRecovery();
      }
      if (!this.recovery) this.restoreRecovery();
      window.__paywallInitialized =
        !!document.getElementById("paywall-warning");
      this.render();
      if (window.__authReady && this.operation.kind === "awaiting-signin") {
        this.authChanged(window.Clerk ?? {});
      }
    }

    private currentForm(): HTMLFormElement | null {
      return document.querySelector<HTMLFormElement>("form[hx-get]");
    }

    private setupClerkListener(): void {
      if (this.unsubscribeClerk || !window.Clerk?.addListener) return;
      window.__paywallClerkListenerRegistered = true;
      this.unsubscribeClerk = window.Clerk.addListener((resources) =>
        this.authChanged(resources),
      );
    }

    private authChanged(resources: ClerkListenerPayload): void {
      if (!resources.user) {
        this.wasUsable = false;
        if (this.recovery) this.stopRecovery();
        if (this.operation.kind === "creating")
          this.operation.controller.abort();
        return;
      }
      if (!resources.session) return;
      const becameUsable = !this.wasUsable;
      this.wasUsable = true;
      if (this.operation.kind === "awaiting-signin") {
        void this.createCheckout(
          resources.user,
          resources.session,
          this.operation.returnUrl,
        );
      } else if (
        becameUsable &&
        this.operation.kind === "idle" &&
        !this.recovery &&
        document.getElementById("paywall-warning")
      ) {
        ClerkHelpers.submitForm();
      }
    }

    private handleClick(event: MouseEvent): void {
      if (!(event.target instanceof Element)) return;
      const button = event.target.closest("button, a");
      if (!button || (button instanceof HTMLButtonElement && button.disabled))
        return;
      if (
        button.id === "clerk-sign-in-button-desktop" ||
        button.id === "clerk-sign-in-button-mobile" ||
        (button instanceof HTMLAnchorElement &&
          button.closest("#desktop-auth-container, #mobile-auth-container") &&
          new URL(button.href).pathname === "/sign-in")
      ) {
        this.clearUpgrade();
        return;
      }
      if (button.id === "upgrade-button") {
        event.preventDefault();
        if (
          this.recovery ||
          this.operation.kind === "creating" ||
          this.operation.kind === "redirecting"
        )
          return;
        const returnUrl = this.cleanUrl(window.location.href);
        this.operation = { kind: "awaiting-signin", returnUrl };
        this.writeStorage(UPGRADE_INTENT_KEY, returnUrl);
        if (window.Clerk?.user && window.Clerk.session) {
          this.wasUsable = true;
          void this.createCheckout(
            window.Clerk.user,
            window.Clerk.session,
            returnUrl,
          );
        } else ClerkHelpers.openSignIn();
      } else if (button.id === "signin-button") {
        event.preventDefault();
        this.clearUpgrade();
        ClerkHelpers.openSignIn();
      } else if (button.id === "retry-button") {
        event.preventDefault();
        this.clearUpgrade();
        if (this.recovery) {
          if (this.recovery.attempt.kind !== "delayed") return;
          const returnUrl = this.recovery.returnUrl;
          this.stopRecovery();
          this.startRecovery(returnUrl, Date.now() + RECOVERY_DURATION_MS);
          if (this.recovery) void this.requestRecovery(this.recovery);
        } else ClerkHelpers.submitForm();
      }
    }

    private clearUpgrade(): void {
      this.writeStorage(UPGRADE_INTENT_KEY, null);
      if (this.operation.kind === "awaiting-signin")
        this.operation = { kind: "idle" };
    }

    private async createCheckout(
      user: ClerkUser,
      session: ClerkSession,
      returnUrl: string,
    ): Promise<void> {
      if (
        this.operation.kind === "creating" ||
        this.operation.kind === "redirecting" ||
        this.recovery
      )
        return;
      const operation = {
        kind: "creating",
        returnUrl,
        userId: user.id,
        controller: new AbortController(),
      } satisfies CheckoutOperation;
      this.operation = operation;
      this.writeStorage(UPGRADE_INTENT_KEY, null);
      this.render();
      const timeout = window.setTimeout(
        () => operation.controller.abort(),
        30000,
      );
      const cancelled = new Promise<never>((_resolve, reject) => {
        operation.controller.signal.addEventListener(
          "abort",
          () => reject(new DOMException("Checkout cancelled", "AbortError")),
          { once: true },
        );
      });
      try {
        const url = await Promise.race([
          cancelled,
          (async () => {
            const token = await session.getToken();
            if (
              !token ||
              operation.controller.signal.aborted ||
              window.Clerk?.user?.id !== operation.userId
            ) {
              throw new Error("Checkout authentication unavailable");
            }
            const response = await fetch("/api/create-checkout", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
              },
              signal: operation.controller.signal,
              body: JSON.stringify({ return_url: returnUrl }),
            });
            if (!response.ok) throw new Error("Checkout creation failed");
            const data: unknown = await response.json();
            if (
              typeof data !== "object" ||
              data === null ||
              !("url" in data) ||
              typeof data.url !== "string" ||
              !data.url
            ) {
              throw new Error("No checkout URL returned");
            }
            return data.url;
          })(),
        ]);
        if (
          operation.controller.signal.aborted ||
          this.operation !== operation ||
          window.Clerk?.user?.id !== operation.userId
        )
          return;
        this.operation = { kind: "redirecting" };
        redirectToPaywallUrl(url);
      } catch (error: unknown) {
        if (this.operation !== operation) return;
        console.error("Upgrade failed:", error);
        this.operation = { kind: "error" };
        this.render();
        window.setTimeout(() => {
          if (this.operation.kind === "error") {
            this.operation = { kind: "idle" };
            this.render();
          }
        }, 3000);
      } finally {
        window.clearTimeout(timeout);
      }
    }

    private restoreRecovery(): void {
      const hint = this.returnHint ?? window.__checkoutReturnUrl;
      this.returnHint = undefined;
      delete window.__checkoutReturnUrl;
      const stored = this.readStorage(RECOVERY_KEY);
      if (stored) {
        try {
          const record: unknown = JSON.parse(stored);
          if (
            typeof record === "object" &&
            record !== null &&
            "returnUrl" in record &&
            "deadline" in record &&
            typeof record.returnUrl === "string" &&
            typeof record.deadline === "number" &&
            Number.isFinite(record.deadline) &&
            this.matchesRoute(record.returnUrl)
          ) {
            this.startRecovery(record.returnUrl, record.deadline);
            return;
          }
        } catch {}
        this.writeStorage(RECOVERY_KEY, null);
      }
      if (hint && this.matchesRoute(hint))
        this.startRecovery(
          this.cleanUrl(hint),
          Date.now() + RECOVERY_DURATION_MS,
        );
    }

    private startRecovery(returnUrl: string, deadline: number): void {
      const form = this.currentForm();
      if (!form) return;
      const recovery: Recovery = {
        returnUrl,
        deadline,
        form,
        deadlineTimer: 0,
        attempt: { kind: "initial" },
      };
      this.recovery = recovery;
      this.clearUpgrade();
      this.writeStorage(RECOVERY_KEY, JSON.stringify({ returnUrl, deadline }));
      recovery.deadlineTimer = window.setTimeout(
        () => this.expireRecovery(recovery),
        Math.max(0, deadline - Date.now()),
      );
      if (deadline <= Date.now()) this.expireRecovery(recovery);
      this.render();
    }

    private configureRecovery(event: HtmxConfigRequestEvent): void {
      const ctx = event.detail.ctx;
      const actorController =
        ctx.sourceEvent && this.retryControllers.get(ctx.sourceEvent);
      if (actorController) {
        const abort = ctx.request.abort;
        ctx.request.signal = actorController.signal;
        ctx.request.abort = () => {
          actorController.abort();
          abort?.();
        };
        if (actorController.signal.aborted) {
          event.preventDefault();
          return;
        }
      }
      const recovery = this.recovery;
      if (!recovery || ctx.sourceElement !== recovery.form) return;
      if (
        recovery.form !== this.currentForm() ||
        !this.matchesRoute(recovery.returnUrl)
      ) {
        this.stopRecovery();
        return;
      }
      const attempt = recovery.attempt;
      if (
        attempt.kind === "initial" ||
        (attempt.kind === "starting" && attempt.event === ctx.sourceEvent)
      ) {
        recovery.attempt = {
          kind: "requesting",
          ctx,
          abort: () => ctx.request.abort?.(),
          recovered: false,
          initial: attempt.kind === "initial",
        };
      } else if (!(attempt.kind === "requesting" && attempt.ctx === ctx))
        this.stopRecovery();
    }

    private scheduleRecovery(recovery: Recovery): void {
      if (this.recovery !== recovery) return;
      if (Date.now() >= recovery.deadline) {
        this.expireRecovery(recovery);
        return;
      }
      recovery.attempt = {
        kind: "waiting",
        timer: window.setTimeout(
          () => void this.requestRecovery(recovery),
          Math.min(RECOVERY_INTERVAL_MS, recovery.deadline - Date.now()),
        ),
      };
      this.render();
    }

    private async requestRecovery(recovery: Recovery): Promise<void> {
      if (
        this.recovery !== recovery ||
        recovery.form !== this.currentForm() ||
        !this.matchesRoute(recovery.returnUrl)
      ) {
        if (this.recovery === recovery) this.stopRecovery();
        return;
      }
      if (Date.now() >= recovery.deadline) {
        this.expireRecovery(recovery);
        return;
      }
      const controller = new AbortController();
      const event = new Event("checkout-recovery");
      const attempt = {
        kind: "starting",
        event,
        controller,
      } satisfies RecoveryAttempt;
      recovery.attempt = attempt;
      this.retryControllers.set(event, controller);
      try {
        await window.htmx?.ajax(
          "GET",
          recovery.form.getAttribute("hx-get") ?? "",
          {
            source: recovery.form,
            target:
              recovery.form.getAttribute("hx-target") ?? "#results-container",
            event,
            values: { push_url: "false" },
          },
        );
      } catch {
      } finally {
        if (this.recovery === recovery && recovery.attempt === attempt)
          this.scheduleRecovery(recovery);
      }
    }

    private expireRecovery(recovery: Recovery): void {
      if (this.recovery !== recovery) return;
      this.abortAttempt(recovery.attempt);
      recovery.attempt = { kind: "delayed" };
      this.render();
    }

    private abortAttempt(attempt: RecoveryAttempt): void {
      if (attempt.kind === "requesting") attempt.abort();
      else if (attempt.kind === "starting") attempt.controller.abort();
      else if (attempt.kind === "waiting") window.clearTimeout(attempt.timer);
    }

    private stopRecovery(keepStored = false): void {
      const recovery = this.recovery;
      this.recovery = null;
      if (recovery) {
        window.clearTimeout(recovery.deadlineTimer);
        this.abortAttempt(recovery.attempt);
      }
      if (!keepStored) this.writeStorage(RECOVERY_KEY, null);
      for (const id of ["upgrade-button", "signin-button", "paywall-buttons"]) {
        const element = document.getElementById(id);
        if (element) element.hidden = false;
      }
      const retry = document.getElementById("retry-button");
      if (retry instanceof HTMLButtonElement) retry.disabled = false;
    }

    private handleEdit(event: Event): void {
      if (
        event.target instanceof Node &&
        this.recovery?.form.contains(event.target)
      )
        this.stopRecovery();
    }

    private suspend(): void {
      this.stopRecovery(true);
      if (this.operation.kind === "creating") {
        this.operation.controller.abort();
        this.operation = { kind: "idle" };
      }
      if (
        this.operation.kind === "awaiting-signin" &&
        !this.matchesRoute(this.operation.returnUrl)
      )
        this.clearUpgrade();
    }

    private render(): void {
      if (this.recovery) {
        if (
          this.recovery.attempt.kind === "requesting" &&
          this.recovery.attempt.recovered
        )
          return;
        const target = document.getElementById("results-container");
        let warning = document.getElementById("paywall-warning");
        if (!warning && target) {
          warning = document.createElement("div");
          warning.id = "paywall-warning";
          warning.className =
            "bg-white border border-sky-200 rounded-lg p-6 shadow text-center";
          warning.innerHTML =
            '<h3 class="text-lg font-semibold text-gray-800"></h3><p class="text-gray-600 my-4"></p><button type="button" id="retry-button" class="text-sky-600 hover:underline">Try again</button>';
          target.replaceChildren(warning);
        }
        const delayed = this.recovery.attempt.kind === "delayed";
        const heading = warning?.querySelector("h3");
        if (heading) heading.textContent = "Payment activation is pending";
        const message = warning?.querySelector("p");
        if (message) {
          message.setAttribute("role", "status");
          message.textContent = delayed
            ? "Payment activation is still pending. Try again."
            : "Payment activation is pending. Checking automatically...";
        }
        for (const id of [
          "upgrade-button",
          "signin-button",
          "paywall-buttons",
        ]) {
          const element = document.getElementById(id);
          if (element) element.hidden = true;
        }
        const retry = document.getElementById("retry-button");
        if (retry instanceof HTMLButtonElement) retry.disabled = !delayed;
        return;
      }
      const button = document.getElementById("upgrade-button");
      if (!(button instanceof HTMLButtonElement)) return;
      button.dataset.upgradeMarkup ??= button.innerHTML;
      button.disabled =
        this.operation.kind === "creating" ||
        this.operation.kind === "redirecting";
      button.innerHTML = button.disabled
        ? "Preparing..."
        : this.operation.kind === "error"
          ? "Error - Try again"
          : button.dataset.upgradeMarkup;
    }

    private cleanUrl(value: string): string {
      const url = new URL(value, window.location.href);
      url.searchParams.delete("checkout_token");
      url.searchParams.delete("customer_session_token");
      return url.href;
    }

    private matchesRoute(value: string): boolean {
      try {
        const retained = new URL(value);
        const current = new URL(window.location.href);
        for (const url of [retained, current]) {
          for (const parameter of [
            "checkout",
            "checkout_token",
            "customer_session_token",
          ])
            url.searchParams.delete(parameter);
          url.searchParams.sort();
        }
        return (
          retained.origin === current.origin &&
          retained.pathname === current.pathname &&
          retained.search === current.search
        );
      } catch {
        return false;
      }
    }

    private readStorage(key: string): string | null {
      try {
        return sessionStorage.getItem(key);
      } catch {
        return null;
      }
    }

    private writeStorage(key: string, value: string | null): void {
      try {
        if (value === null) sessionStorage.removeItem(key);
        else sessionStorage.setItem(key, value);
      } catch {}
    }
  }

  const manager = new PaywallManager();
  window.__initPaywall = () => manager.refresh();
  document.addEventListener("DOMContentLoaded", () => manager.refresh());
}
