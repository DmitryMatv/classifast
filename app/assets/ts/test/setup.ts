import { afterEach, beforeEach, vi } from "vitest";

function createMatchMediaMock(): typeof window.matchMedia {
  return vi.fn((query: string): MediaQueryList => {
    return {
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(() => false),
    };
  });
}

function createHtmxMock(): HtmxInstance {
  return {
    trigger: vi.fn(),
    process: vi.fn(),
    ajax: vi.fn(async () => {}),
  };
}

function createClerkMock(): ClerkInstance {
  return {
    load: vi.fn(async () => {}),
    addListener: vi.fn(() => vi.fn()),
    mountUserButton: vi.fn(),
    openSignIn: vi.fn(),
    openSignUp: vi.fn(),
    openGoogleOneTap: vi.fn(),
  };
}

beforeEach(() => {
  document.body.innerHTML = "";
  document.head.innerHTML = "";
  document.documentElement.className = "";

  delete window.__authReady;
  delete window.__clerkAuthListenerRegistered;
  delete window.__clerkInteractionListenersRegistered;
  delete window.__initPaywall;
  delete window.__paywallClerkListenerRegistered;
  delete window.__paywallInitialized;
  delete window.__paywallNavigate;
  delete window.__paywallScriptParsed;
  delete window.ShareLink;
  delete window.copyOriginalId;

  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: createMatchMediaMock(),
  });

  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    writable: true,
    value: vi.fn((callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    }),
  });

  vi.stubGlobal("requestAnimationFrame", window.requestAnimationFrame);

  Object.defineProperty(window, "scrollTo", {
    configurable: true,
    writable: true,
    value: vi.fn(),
  });

  Object.defineProperty(document, "execCommand", {
    configurable: true,
    writable: true,
    value: vi.fn(() => true),
  });

  Object.defineProperty(HTMLFormElement.prototype, "requestSubmit", {
    configurable: true,
    writable: true,
    value: vi.fn(),
  });

  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: {
      writeText: vi.fn(async () => {}),
    },
  });

  window.htmx = createHtmxMock();
  window.Clerk = createClerkMock();

  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      throw new Error("Unexpected fetch call");
    }),
  );
});

afterEach(() => {
  vi.clearAllMocks();
  vi.clearAllTimers();
  vi.unstubAllGlobals();
});
