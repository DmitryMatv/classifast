// Simple Clerk Authentication using React-like components
(function () {
    'use strict';

    // Helper: wait for an element to appear in the DOM using MutationObserver with timeout
    const DEFAULT_WAIT_MS = 100;
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

            // Immediate check if Clerk is already available
            if (typeof window.Clerk !== 'undefined') {
                console.log('✅ Clerk already available');
                await this.initializeClerk();
                return;
            }

            // Wait for DOM to be ready first (faster than waiting for both)
            if (document.readyState !== 'loading') {
                console.log('✅ DOM already ready');
                await this.waitForClerk();
            } else {
                document.addEventListener('DOMContentLoaded', () => {
                    console.log('✅ DOM ready event fired');
                    this.waitForClerk();
                });
            }
        },

        async waitForClerk() {
            return new Promise((resolve, reject) => {
                const clerkCheckInterval = setInterval(() => {
                    if (typeof window.Clerk !== 'undefined') {
                        clearInterval(clerkCheckInterval);
                        console.log('✅ Clerk object found:', window.Clerk);
                        this.initializeClerk().then(resolve).catch(reject);
                    }
                }, 50); // Reduced from 100ms to 50ms for faster detection

                // Timeout after 5 seconds (reduced from 10 seconds)
                setTimeout(() => {
                    clearInterval(clerkCheckInterval);
                    if (typeof window.Clerk === 'undefined') {
                        console.error('❌ Clerk failed to load within 5 seconds');
                        reject(new Error('Clerk failed to load within 5 seconds'));
                    }
                }, 5000);
            });
        },

        async initializeClerk() {
            try {
                console.log('✅ Clerk available:', typeof window.Clerk !== 'undefined');
                console.log('📋 Clerk version:', window.Clerk.version || 'unknown');

                // Initialize Clerk
                if (window.Clerk.load) {
                    await window.Clerk.load();
                    console.log('✅ Clerk loaded successfully');
                } else {
                    console.log('⚠️ Clerk.load method not found, assuming preloaded');
                }

                // Check authentication state and render appropriate UI immediately
                this.checkAuthAndRender();

                // Set up proper event listeners for auth state changes
                this.setupAuthListeners();

                console.log('✅ Clerk auth initialized successfully');
            } catch (error) {
                console.error('❌ Clerk auth initialization failed:', error);
                this.renderFallbackAuth();
                throw error;
            }
        },

        async checkAuthAndRender() {
            // Use proper Clerk.isSignedIn property instead of checking for user object
            const isSignedIn = window.Clerk && window.Clerk.isSignedIn ? window.Clerk.isSignedIn : false;
            console.log('🔐 User signed in:', isSignedIn);
            console.log('👤 User object:', window.Clerk?.user);

            // Clear containers before rendering to prevent duplication
            this.clearAuthContainers();

            if (isSignedIn) {
                this.renderSignedIn();
            } else {
                this.renderSignedOut();
            }

            this.renderGoogleOneTap();
        },

        clearAuthContainers() {
            const desktopContainer = document.getElementById('desktop-auth-container');
            const mobileContainer = document.getElementById('mobile-auth-container');

            if (desktopContainer) {
                desktopContainer.innerHTML = '';
            }
            if (mobileContainer) {
                mobileContainer.innerHTML = '';
            }
        },

        setupAuthListeners() {
            // Set up proper event listeners for auth state changes
            if (window.Clerk.addListener) {
                // Clear any existing poller if a real listener is now available
                this.clearAuthPoll();

                // Listen for various auth events
                window.Clerk.addListener((event) => {
                    console.log('🔄 Auth state changed:', event);
                    // Force UI update immediately on any auth state change
                    setTimeout(() => this.checkAuthAndRender(), 10);
                });

                // Also add listeners for specific events if available
                if (window.Clerk.on) {
                    window.Clerk.on('signedIn', () => {
                        console.log('✅ User signed in event');
                        this.checkAuthAndRender();
                    });

                    window.Clerk.on('signedOut', () => {
                        console.log('🔓 User signed out event');
                        this.checkAuthAndRender();
                    });
                }
            } else {
                console.log('⚠️ Clerk.addListener not available, falling back to polling');
                // Fallback: Poll every 3 seconds for auth state changes (with guard to prevent duplicate pollers)
                if (!this.authPollInterval) {
                    this.authPollInterval = setInterval(() => this.checkAuthAndRender(), 3000);
                    console.log('📡 Auth polling started (3s interval)');
                }
            }
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

            // Create sign-in button for signed-out users

            // Create user button for signed-in users
            const userButtonHTML = '<div id="clerk-user-button-desktop" class="clerk-user-button"></div>';
            const mobileUserButtonHTML = '<div id="clerk-user-button-mobile" class="clerk-user-button"></div>';

            if (desktopContainer) {
                desktopContainer.innerHTML = userButtonHTML;
                console.log('📱 Desktop user button container created');

                // Wait for the inserted element with shorter timeout
                waitForElement('clerk-user-button-desktop', 50)
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
                                // Retry once after a short delay
                                setTimeout(() => this.renderSignedIn(), 1000);
                            }
                        } else {
                            console.error('❌ Failed to mount desktop user button - element or method not found');
                        }
                    })
                    .catch((err) => {
                        console.error('❌ Timeout or error waiting for desktop user button:', err);
                        // Retry once after a short delay
                        setTimeout(() => this.renderSignedIn(), 1000);
                    });
            }

            if (mobileContainer) {
                mobileContainer.innerHTML = mobileUserButtonHTML;
                console.log('📱 Mobile user button container created');

                waitForElement('clerk-user-button-mobile', 50)
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
                                // Retry once after a short delay
                                setTimeout(() => this.renderSignedIn(), 1000);
                            }
                        } else {
                            console.error('❌ Failed to mount mobile user button - element or method not found');
                        }
                    })
                    .catch((err) => {
                        console.error('❌ Timeout or error waiting for mobile user button:', err);
                        // Retry once after a short delay
                        setTimeout(() => this.renderSignedIn(), 1000);
                    });
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

            // Clear any existing content first
            if (desktopContainer) {
                desktopContainer.innerHTML = '';
            }
            if (mobileContainer) {
                mobileContainer.innerHTML = '';
            }

            // Create sign-in button for signed-out users
            const signInButtonHTML = '<div id="clerk-sign-in-button-desktop" class="bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform cursor-pointer">Sign In</div>';
            const mobileSignInButtonHTML = '<div id="clerk-sign-in-button-mobile" class="bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform cursor-pointer w-full text-center mb-2">Sign In</div>';

            if (desktopContainer) {
                desktopContainer.innerHTML = signInButtonHTML;
                console.log('📱 Desktop sign-in button created');

                // Immediately attach click handler since element is in DOM
                const desktopButton = document.getElementById('clerk-sign-in-button-desktop');
                if (desktopButton) {
                    this.attachSignInHandler(desktopButton, 'Desktop');
                } else {
                    // Fallback to waitForElement if immediate access fails
                    waitForElement('clerk-sign-in-button-desktop', 50)
                        .then((button) => {
                            if (button) {
                                this.attachSignInHandler(button, 'Desktop');
                            } else {
                                console.error('❌ Desktop sign-in button element not found');
                            }
                        })
                        .catch((err) => console.error('❌ Timeout waiting for desktop sign-in button:', err));
                }
            }

            if (mobileContainer) {
                mobileContainer.innerHTML = mobileSignInButtonHTML;
                console.log('📱 Mobile sign-in button created');

                // Immediately attach click handler since element is in DOM
                const mobileButton = document.getElementById('clerk-sign-in-button-mobile');
                if (mobileButton) {
                    this.attachSignInHandler(mobileButton, 'Mobile');
                } else {
                    // Fallback to waitForElement if immediate access fails
                    waitForElement('clerk-sign-in-button-mobile', 50)
                        .then((button) => {
                            if (button) {
                                this.attachSignInHandler(button, 'Mobile');
                            } else {
                                console.error('❌ Mobile sign-in button element not found');
                            }
                        })
                        .catch((err) => console.error('❌ Timeout waiting for mobile sign-in button:', err));
                }
            }
        },

        attachSignInHandler(button, type) {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                console.log(`👆 ${type} sign-in button clicked`);
                if (window.Clerk && window.Clerk.openSignIn) {
                    window.Clerk.openSignIn();
                    console.log('✅ Opening Clerk sign-in modal');
                } else {
                    console.error('❌ Clerk openSignIn method not available');
                    // Fallback: redirect to sign-in page
                    window.location.href = 'https://accounts.classifast.com/sign-in?redirect_url=' + encodeURIComponent(window.location.href);
                }
            });
            console.log(`✅ ${type} sign-in handler attached`);
        },

        renderGoogleOneTap() {
            console.log('🎯 Attempting to open Google One Tap...');

            if (typeof window.Clerk === 'undefined') {
                console.error('❌ Clerk not available for Google One Tap');
                return;
            }

            try {
                // Check if user is signed in - don't show Google One Tap if already signed in
                if (window.Clerk.isSignedIn) {
                    console.log('ℹ️ User already signed in, skipping Google One Tap');
                    return;
                }

                // Open Google One Tap with proper configuration
                if (window.Clerk.openGoogleOneTap) {
                    const params = {
                        cancelOnTapOutside: false,
                        itpSupport: true,
                        fedCmSupport: true
                    };

                    window.Clerk.openGoogleOneTap(params);
                    console.log('✅ Google One Tap opened successfully with params:', params);
                } else {
                    console.log('⚠️ Google One Tap openGoogleOneTap method not available in this Clerk version');
                }
            } catch (error) {
                console.error('❌ Google One Tap failed to open:', error);
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
