// Simple Clerk Authentication using React-like components
(function () {
    'use strict';

    // Helper: wait for an element to appear in the DOM using MutationObserver with timeout
    const DEFAULT_WAIT_MS = 3000;
    function waitForElement(idOrSelector, timeoutMs = DEFAULT_WAIT_MS) {
        return new Promise((resolve, reject) => {
            // finder tries getElementById first (caller often passes id without '#'), then querySelector
            const find = () => {
                if (!idOrSelector) return null;
                let el = null;
                try {
                    el = document.getElementById(idOrSelector);
                    if (el) return el;
                } catch (e) {
                    // ignore and try querySelector
                }
                try {
                    return document.querySelector(idOrSelector);
                } catch (e) {
                    return null;
                }
            };

            const existing = find();
            if (existing) {
                return resolve(existing);
            }

            const observer = new MutationObserver((mutations, obs) => {
                const el = find();
                if (el) {
                    try { obs.disconnect(); } catch (e) { /* ignore */ }
                    clearTimeout(timer);
                    resolve(el);
                }
            });

            // Observe the document for changes
            observer.observe(document.documentElement || document.body, { childList: true, subtree: true });

            const timer = setTimeout(() => {
                try { observer.disconnect(); } catch (e) { /* ignore */ }
                reject(new Error('Timeout waiting for element: ' + idOrSelector));
            }, timeoutMs);
        });
    }

    window.ClerkAuth = {
        async init() {
            console.log('🚀 Initializing simple Clerk auth...');

            // Wait for both DOM and Clerk to be ready
            const waitForClerkAndDOM = () => {
                return new Promise((resolve, reject) => {
                    let clerkReady = false;
                    let domReady = false;

                    const checkComplete = () => {
                        console.log('📊 Status check - Clerk ready:', clerkReady, 'DOM ready:', domReady);
                        if (clerkReady && domReady) {
                            resolve();
                        }
                    };

                    // Check for DOM
                    if (document.readyState !== 'loading') {
                        domReady = true;
                        console.log('✅ DOM already ready');
                    } else {
                        document.addEventListener('DOMContentLoaded', () => {
                            domReady = true;
                            console.log('✅ DOM ready event fired');
                            checkComplete();
                        });
                    }

                    // Check for Clerk with timeout
                    const clerkCheckInterval = setInterval(() => {
                        if (typeof window.Clerk !== 'undefined') {
                            clerkReady = true;
                            console.log('✅ Clerk object found:', window.Clerk);
                            clearInterval(clerkCheckInterval);
                            checkComplete();
                        }
                    }, 100);

                    // Timeout after 10 seconds
                    setTimeout(() => {
                        clearInterval(clerkCheckInterval);
                        if (!clerkReady) {
                            console.error('❌ Clerk failed to load within 10 seconds');
                            reject(new Error('Clerk failed to load within 10 seconds'));
                        }
                    }, 10000);
                });
            };

            try {
                await waitForClerkAndDOM();
                console.log('✅ Clerk available:', typeof window.Clerk !== 'undefined');
                console.log('📋 Clerk version:', window.Clerk.version || 'unknown');

                // Initialize Clerk
                if (window.Clerk.load) {
                    await window.Clerk.load();
                    console.log('✅ Clerk loaded successfully');
                } else {
                    console.log('⚠️ Clerk.load method not found, assuming preloaded');
                }

                // Check authentication state and render appropriate UI
                this.checkAuthAndRender();

                // Listen for auth state changes
                if (window.Clerk.addListener) {
                    // Clear any existing poller if a real listener is now available
                    this.clearAuthPoll();
                    window.Clerk.addListener((event) => {
                        console.log('🔄 Auth state changed:', event);
                        this.checkAuthAndRender();
                    });
                } else {
                    console.log('⚠️ Clerk.addListener not available, falling back to polling');
                    // Fallback: Poll every 10 seconds for auth state changes (with guard to prevent duplicate pollers)
                    if (!this.authPollInterval) {
                        this.authPollInterval = setInterval(() => this.checkAuthAndRender(), 10000);
                        console.log('📡 Auth polling started (10s interval)');
                    }
                }

                console.log('✅ Clerk auth initialized successfully');
            } catch (error) {
                console.error('❌ Clerk auth initialization failed:', error);
                this.renderFallbackAuth();
            }
        },

        async checkAuthAndRender() {
            // Use proper Clerk.isSignedIn property instead of checking for user object
            const isSignedIn = window.Clerk && window.Clerk.isSignedIn ? window.Clerk.isSignedIn : false;
            console.log('🔐 User signed in:', isSignedIn);
            console.log('👤 User object:', window.Clerk?.user);

            if (isSignedIn) {
                this.renderSignedIn();
            } else {
                this.renderSignedOut();
            }

            this.renderGoogleOneTap();
        },

        clearAuthPoll() {
            if (this.authPollInterval) {
                clearInterval(this.authPollInterval);
                this.authPollInterval = null;
                console.log('🛑 Auth polling stopped');
            }
        },

        renderSignedIn() {
            console.log('🎨 Rendering signed-in state...');

            const desktopContainer = document.getElementById('desktop-auth-container');
            const mobileContainer = document.getElementById('mobile-auth-container');

            if (!desktopContainer && !mobileContainer) {
                console.error('❌ Auth containers not found');
                return;
            }

            // Create user button for signed-in users
            const userButtonHTML = '<div id="clerk-user-button-desktop" class="clerk-user-button"></div>';
            const mobileUserButtonHTML = '<div id="clerk-user-button-mobile" class="clerk-user-button"></div>';

            if (desktopContainer) {
                desktopContainer.innerHTML = userButtonHTML;
                console.log('📱 Desktop user button container created');

                // Wait for the inserted element instead of using a fixed timeout
                waitForElement('clerk-user-button-desktop', DEFAULT_WAIT_MS)
                    .then((userButtonEl) => {
                        if (userButtonEl && window.Clerk && window.Clerk.mountUserButton) {
                            try {
                                window.Clerk.mountUserButton(userButtonEl, {
                                    appearance: {
                                        elements: {
                                            userButtonAvatarBox: 'w-8 h-8',
                                            userButtonBox: 'h-8'
                                        }
                                    }
                                });
                                console.log('✅ Desktop user button mounted successfully');
                            } catch (err) {
                                console.error('❌ Error mounting desktop user button:', err);
                            }
                        } else {
                            console.error('❌ Failed to mount desktop user button - element or method not found');
                        }
                    })
                    .catch((err) => console.error('❌ Timeout or error waiting for desktop user button:', err));
            }

            if (mobileContainer) {
                mobileContainer.innerHTML = mobileUserButtonHTML;
                console.log('📱 Mobile user button container created');

                waitForElement('clerk-user-button-mobile', DEFAULT_WAIT_MS)
                    .then((userButtonEl) => {
                        if (userButtonEl && window.Clerk && window.Clerk.mountUserButton) {
                            try {
                                window.Clerk.mountUserButton(userButtonEl, {
                                    appearance: {
                                        elements: {
                                            userButtonAvatarBox: 'w-8 h-8',
                                            userButtonBox: 'h-8'
                                        }
                                    }
                                });
                                console.log('✅ Mobile user button mounted successfully');
                            } catch (err) {
                                console.error('❌ Error mounting mobile user button:', err);
                            }
                        } else {
                            console.error('❌ Failed to mount mobile user button - element or method not found');
                        }
                    })
                    .catch((err) => console.error('❌ Timeout or error waiting for mobile user button:', err));
            }
        },

        renderSignedOut() {
            console.log('🔓 Rendering signed-out state...');

            const desktopContainer = document.getElementById('desktop-auth-container');
            const mobileContainer = document.getElementById('mobile-auth-container');

            if (!desktopContainer && !mobileContainer) {
                console.error('❌ Auth containers not found');
                return;
            }

            // Create sign-in button for signed-out users
            const signInButtonHTML = '<div id="clerk-sign-in-button-desktop" class="bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform cursor-pointer">Sign In</div>';
            const mobileSignInButtonHTML = '<div id="clerk-sign-in-button-mobile" class="bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform cursor-pointer w-full text-center mb-2">Sign In</div>';

            if (desktopContainer) {
                desktopContainer.innerHTML = signInButtonHTML;
                console.log('📱 Desktop sign-in button created');

                waitForElement('clerk-sign-in-button-desktop', DEFAULT_WAIT_MS)
                    .then((button) => {
                        if (button) {
                            button.addEventListener('click', (e) => {
                                e.preventDefault();
                                console.log('👆 Desktop sign-in button clicked');
                                if (window.Clerk && window.Clerk.openSignIn) {
                                    window.Clerk.openSignIn();
                                    console.log('✅ Opening Clerk sign-in modal');
                                } else {
                                    console.error('❌ Clerk openSignIn method not available');
                                    // Fallback: redirect to sign-in page
                                    window.location.href = 'https://accounts.classifast.com/sign-in?redirect_url=' + encodeURIComponent(window.location.href);
                                }
                            });
                            console.log('✅ Desktop sign-in handler attached');
                        } else {
                            console.error('❌ Desktop sign-in button element not found');
                        }
                    })
                    .catch((err) => console.error('❌ Timeout or error waiting for desktop sign-in button:', err));
            }

            if (mobileContainer) {
                mobileContainer.innerHTML = mobileSignInButtonHTML;
                console.log('📱 Mobile sign-in button created');

                waitForElement('clerk-sign-in-button-mobile', DEFAULT_WAIT_MS)
                    .then((button) => {
                        if (button) {
                            button.addEventListener('click', (e) => {
                                e.preventDefault();
                                console.log('👆 Mobile sign-in button clicked');
                                if (window.Clerk && window.Clerk.openSignIn) {
                                    window.Clerk.openSignIn();
                                    console.log('✅ Opening Clerk sign-in modal');
                                } else {
                                    console.error('❌ Clerk openSignIn method not available');
                                    // Fallback: redirect to sign-in page
                                    window.location.href = 'https://accounts.classifast.com/sign-in?redirect_url=' + encodeURIComponent(window.location.href);
                                }
                            });
                            console.log('✅ Mobile sign-in handler attached');
                        } else {
                            console.error('❌ Mobile sign-in button element not found');
                        }
                    })
                    .catch((err) => console.error('❌ Timeout or error waiting for mobile sign-in button:', err));
            }
        },

        renderGoogleOneTap() {
            console.log('🎯 Attempting to mount Google One Tap...');
            const googleOneTapContainer = document.getElementById('google-one-tap');

            if (!googleOneTapContainer) {
                console.log('⚠️ Google One Tap container not found');
                return;
            }

            if (typeof window.Clerk === 'undefined') {
                console.error('❌ Clerk not available for Google One Tap');
                return;
            }

            try {
                // Clear any existing content
                googleOneTapContainer.innerHTML = '';

                // Check if user is signed in - don't show Google One Tap if already signed in
                if (window.Clerk.isSignedIn) {
                    console.log('ℹ️ User already signed in, skipping Google One Tap');
                    return;
                }

                // Mount Google One Tap
                if (window.Clerk.mountGoogleOneTap) {
                    window.Clerk.mountGoogleOneTap(googleOneTapContainer);
                    console.log('✅ Google One Tap mounted successfully');
                } else {
                    console.log('⚠️ Google One Tap method not available in this Clerk version');
                }
            } catch (error) {
                console.error('❌ Google One Tap failed to mount:', error);
                console.log('📋 Error details:', error.message || error);
            }
        },

        renderFallbackAuth() {
            console.log('🚨 Using fallback authentication UI');
            const desktopContainer = document.getElementById('desktop-auth-container');
            const mobileContainer = document.getElementById('mobile-auth-container');
            const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

            const fallbackHTML = isDevelopment
                ? '<a href="https://accounts.classifast.com/sign-in" class="bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform">Sign In</a>'
                : '<a href="https://accounts.classifast.com/sign-in?redirect_url=' + encodeURIComponent(window.location.href) + '" class="bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform">Sign In</a>';

            if (desktopContainer) {
                desktopContainer.innerHTML = fallbackHTML;
                console.log('📱 Desktop fallback rendered');
            }
            if (mobileContainer) {
                mobileContainer.innerHTML = fallbackHTML;
                console.log('📱 Mobile fallback rendered');
            }
        }
    };

    // Initialize when script loads (no need to wait for DOM since we handle it internally)
    window.ClerkAuth.init();

    // Cleanup on page unload to prevent memory leaks
    window.addEventListener('beforeunload', () => {
        window.ClerkAuth.clearAuthPoll();
    });
})();
