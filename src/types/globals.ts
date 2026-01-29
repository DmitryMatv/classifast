// Type declarations for external libraries used in Classifast

declare global {
  // ============================================
  // Window extensions
  // ============================================
  interface Window {
    Clerk?: ClerkInstance;
    __clerkAuthListenerRegistered?: boolean;
    __paywallClerkListenerRegistered?: boolean;
    copyOriginalId?: (text: string, buttonElement: HTMLButtonElement) => void;
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
    getToken: () => Promise<string | null>;
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
    };
  }

  interface HTMLElementEventMap {
    "htmx:configRequest": HtmxConfigRequestEvent;
    "htmx:authReady": CustomEvent;
    "clerk:loaded": CustomEvent;
  }
}

export {};
