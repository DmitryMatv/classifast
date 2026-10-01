// Type declarations for external libraries used in Classifast

declare global {
  // ============================================
  // Window extensions
  // ============================================
  interface Window {
    Clerk?: ClerkInstance;
    __authReady?: boolean;
    __checkoutReturnUrl?: string;
    __commonLifecycleAbort?: AbortController;
    __commonController?: {
      init: () => void;
      getCachedAuthToken: () => string | null;
      refreshAuthToken: () => Promise<string | null>;
      abort: AbortController;
    };
    __classifierLifecycleAbort?: AbortController;
    __classifierHistoryAbort?: AbortController;
    __clerkAuthListenerRegistered?: boolean;
    __clerkInteractionListenersRegistered?: boolean;
    __clerkScriptFailed?: boolean;
    __internal_ClerkUICtor?: unknown;
    __initPaywall?: () => void;
    __paywallClerkListenerRegistered?: boolean;
    __paywallInitialized?: boolean;
    __paywallNavigate?: (url: string) => void;
    __paywallScriptParsed?: boolean;
    __storefrontInitialized?: boolean;
    __storefrontNavigate?: (url: string) => void;
    htmx?: HtmxInstance;
    ShareLink?: {
      copyShareableLink: () => Promise<void>;
    };
  }

  // ============================================
  // HTMX Global Instance
  // ============================================
  interface HtmxInstance {
    trigger: (element: Element | string, event: string) => void;
    process: (element: Element) => void;
    ajax: (
      method: string,
      url: string,
      options: {
        source?: Element;
        event?: Event;
        values?: Record<string, string>;
        headers?: Record<string, string>;
        select?: string;
        push?: boolean | string;
        request?: Partial<HtmxRequestContext["request"]>;
        target?: string | Element;
        swap?: string;
      },
    ) => Promise<void>;
  }

  // ============================================
  // Clerk Authentication SDK
  // ============================================
  interface ClerkLoadOptions {
    ui?: {
      ClerkUI: NonNullable<Window["__internal_ClerkUICtor"]>;
    };
  }

  interface ClerkInstance {
    load: (options?: ClerkLoadOptions) => Promise<void>;
    session?: ClerkSession;
    user?: ClerkUser;
    addListener: (
      callback: (payload: ClerkListenerPayload) => void,
    ) => () => void;
    mountUserButton: (
      element: HTMLElement,
      options?: UserButtonOptions,
    ) => void;
    openSignIn: (options?: SignInOptions) => void;
    openSignUp: (options?: SignUpOptions) => void;
    openGoogleOneTap: (params?: GoogleOneTapParams) => void;
  }

  interface ClerkSession {
    getToken: (options?: {
      skipCache?: boolean;
      expirationBufferSeconds?: number;
      template?: string;
    }) => Promise<string | null>;
  }

  interface ClerkUser {
    id: string;
    firstName?: string;
    lastName?: string;
    emailAddresses?: Array<{ emailAddress: string }>;
    imageUrl?: string;
  }

  interface ClerkListenerPayload {
    user?: ClerkUser;
    session?: ClerkSession;
  }

  interface UserButtonOptions {
    appearance?: {
      elements?: {
        userButtonAvatarBox?: string;
        userButtonBox?: string;
        userButtonTrigger?: string;
      };
    };
  }

  interface SignInOptions {
    redirectUrl?: string;
  }

  interface SignUpOptions {
    redirectUrl?: string;
  }

  interface GoogleOneTapParams {
    cancelOnTapOutside?: boolean;
    itpSupport?: boolean;
    fedCmSupport?: boolean;
  }

  // ============================================
  // HTMX Extensions (htmx 4 event detail shapes)
  // ============================================
  interface HtmxRequestContext<
    RequestBody extends BodyInit | null = BodyInit | null,
  > {
    sourceElement: Element;
    sourceEvent?: Event;
    target: Element;
    swap?: string;
    select?: string;
    fetch?: typeof window.fetch;
    request: Omit<RequestInit, "body" | "headers" | "method"> & {
      action: string;
      method: string;
      headers: Record<string, string>;
      body: RequestBody;
      abort?: () => void;
      timeout?: number | string;
    };
    response?: {
      status: number;
      headers: Headers;
    };
    text?: string;
  }

  interface HtmxConfigRequestEvent extends CustomEvent {
    detail: { ctx: HtmxRequestContext<FormData> };
  }

  interface HtmxBeforeRequestEvent extends CustomEvent {
    detail: { ctx: HtmxRequestContext };
  }

  interface HtmxAfterRequestEvent extends CustomEvent {
    detail: { ctx: HtmxRequestContext };
  }

  interface HtmxAfterSwapEvent extends CustomEvent {
    detail: { ctx: HtmxRequestContext };
  }

  interface HtmxResponseErrorEvent extends CustomEvent {
    detail: {
      ctx: HtmxRequestContext & {
        response: { status: number; headers: Headers };
      };
    };
  }

  interface HtmxHistoryUpdateEvent extends CustomEvent {
    detail: {
      history?: { type?: string; path?: string };
    };
  }

  interface HTMLElementEventMap {
    "htmx:before:request": HtmxBeforeRequestEvent;
    "htmx:config:request": HtmxConfigRequestEvent;
    "htmx:after:swap": HtmxAfterSwapEvent;
    "htmx:after:request": HtmxAfterRequestEvent;
    "htmx:finally:request": HtmxAfterRequestEvent;
    "htmx:response:error": HtmxResponseErrorEvent;
    "htmx:authReady": CustomEvent;
    "clerk:loaded": CustomEvent;
    "htmx:error": CustomEvent;
    "htmx:before:history:update": HtmxHistoryUpdateEvent;
    "htmx:before:history:restore": CustomEvent;
  }
}

export {};
