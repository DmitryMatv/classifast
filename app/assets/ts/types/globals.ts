// Type declarations for external libraries used in Classifast

declare global {
  // ============================================
  // Window extensions
  // ============================================
  interface Window {
    Clerk?: ClerkInstance;
    __clerkAuthListenerRegistered?: boolean;
    __clerkInteractionListenersRegistered?: boolean;
    __classifierNavigate?: (url: string) => void;
    __initPaywall?: () => void;
    __paywallClerkListenerRegistered?: boolean;
    __paywallInitialized?: boolean;
    __paywallNavigate?: (url: string) => void;
    __paywallScriptParsed?: boolean;
    copyOriginalId?: (text: string, buttonElement: HTMLButtonElement) => void;
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
        target?: string | Element;
        swap?: string;
      },
    ) => Promise<void>;
  }

  // ============================================
  // Clerk Authentication SDK
  // ============================================
  interface ClerkInstance {
    load: () => Promise<void>;
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
  // HTMX Extensions
  // ============================================
  interface HtmxConfigRequestEvent extends CustomEvent {
    detail: {
      headers: Record<string, string>;
      xhr: XMLHttpRequest;
      elt: Element;
    };
  }

  interface HtmxBeforeRequestEvent extends CustomEvent {
    detail: {
      elt: Element;
      target: HTMLElement;
    };
  }

  interface HtmxAfterSwapEvent extends CustomEvent {
    detail: {
      target: HTMLElement;
    };
  }

  interface HtmxAfterSettleEvent extends CustomEvent {
    detail: {
      target: HTMLElement;
    };
  }

  interface HtmxAfterRequestEvent extends CustomEvent {
    detail: {
      elt: Element;
      target: HTMLElement;
    };
  }

  interface HtmxResponseErrorEvent extends CustomEvent {
    detail: {
      xhr: XMLHttpRequest;
      target: HTMLElement;
    };
  }

  interface HTMLElementEventMap {
    "htmx:beforeRequest": HtmxBeforeRequestEvent;
    "htmx:configRequest": HtmxConfigRequestEvent;
    "htmx:afterSwap": HtmxAfterSwapEvent;
    "htmx:afterSettle": HtmxAfterSettleEvent;
    "htmx:afterRequest": HtmxAfterRequestEvent;
    "htmx:responseError": HtmxResponseErrorEvent;
    "htmx:authReady": CustomEvent;
    "clerk:loaded": CustomEvent;
    "htmx:sendAbort": CustomEvent;
    "htmx:timeout": CustomEvent;
    "htmx:beforeHistorySave": CustomEvent;
    "htmx:historyRestore": CustomEvent;
  }
}

export {};
