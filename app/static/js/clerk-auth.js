// Shared Clerk Authentication Module
(function () {
    'use strict';

    // Clerk Authentication Module - Completely rewritten for reliability
    window.ClerkAuth = {

        // Configuration
        config: {
            clerkLoadTimeout: 10000, // 10 seconds timeout
            retryInterval: 100,      // Retry every 100ms
            maxRetries: 100,         // Max 10 seconds of retries
            isDevelopment: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        },

        // Track initialization state
        isInitialized: false,
        clerkLoadRetries: 0,

        setupClerkModalHandling() {
            const checkForClerkModal = () => {
                const modalExists = document.querySelector('.cl-modal') ||
                    document.querySelector('.cl-overlay') ||
                    document.querySelector('[class*="cl-modal"]') ||
                    document.querySelector('[class*="cl-overlay"]') ||
                    document.body.style.overflow === 'hidden';

                if (modalExists) {
                    document.body.classList.add('clerk-modal-open');
                    document.documentElement.classList.add('clerk-modal-open');
                } else {
                    document.body.classList.remove('clerk-modal-open');
                    document.documentElement.classList.remove('clerk-modal-open');
                }
            };

            if ('MutationObserver' in window) {
                const observer = new MutationObserver(checkForClerkModal);
                observer.observe(document.body, {
                    attributes: true,
                    attributeFilter: ['style'],
                    subtree: true,
                    childList: true
                });
            }

            document.addEventListener('click', (e) => {
                if (e.target.id?.includes('login')) {
                    setTimeout(checkForClerkModal, 100);
                }
            });

            setTimeout(checkForClerkModal, 100);
        },

        renderLoggedOutButtons(container, variant) {
            if (!container) return;

            const loginButtonClasses = variant === 'desktop'
                ? 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform'
                : 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform w-full text-center mb-2';

            container.innerHTML = `
                <div class="flex ${variant === 'desktop' ? 'items-center' : 'flex-col'}">
                    <button type="button" id="${variant}-login-button" class="${loginButtonClasses}">
                        Sign In
                    </button>
                </div>
            `;

            console.log(`Rendered login button for ${variant} variant`);
        },

        bindAuthActions(loginId) {
            const loginButton = document.getElementById(loginId);

            if (loginButton) {
                loginButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    if (typeof Clerk !== 'undefined' && Clerk.openSignIn) {
                        console.log('Opening Clerk Sign In');
                        Clerk.openSignIn({
                            afterSignInUrl: window.location.href,
                            afterSignUpUrl: window.location.href
                        });
                    } else {
                        // Fallback to direct URL - use clean URL for localhost, full redirect for production
                        if (this.config.isDevelopment) {
                            window.location.href = 'https://accounts.classifast.com/sign-in';
                        } else {
                            window.location.href = 'https://accounts.classifast.com/sign-in?redirect_url=' + encodeURIComponent(window.location.href);
                        }
                    }
                });
            }
        },

        renderClerkUserButton(container, variant) {
            if (!container) return;

            container.innerHTML = `<div id="${variant}-user-button"></div>`;
            const userButtonDiv = document.getElementById(`${variant}-user-button`);

            try {
                if (typeof Clerk !== 'undefined' && Clerk.mountUserButton) {
                    Clerk.mountUserButton(userButtonDiv, {
                        appearance: {
                            elements: {
                                userButtonAvatarBox: 'w-8 h-8',
                                userButtonBox: 'h-8',
                            }
                        }
                    });
                    console.log(`Mounted Clerk user button for ${variant}`);
                } else {
                    throw new Error('Clerk or mountUserButton not available');
                }
            } catch (error) {
                console.error('Failed to mount Clerk user button:', error);
                // Fallback to simple avatar display
                if (Clerk.user) {
                    container.innerHTML = `
                        <div class="flex items-center space-x-2">
                            <div class="w-8 h-8 rounded-full bg-sky-600 flex items-center justify-center text-white font-semibold">
                                ${Clerk.user.firstName?.charAt(0) || Clerk.user.email?.charAt(0) || 'U'}
                            </div>
                            <button onclick="if(typeof Clerk!=='undefined' && Clerk.signOut) Clerk.signOut()" class="text-gray-600 hover:text-sky-600">
                                Sign Out
                            </button>
                        </div>
                    `;
                }
            }
        },

        renderFallbackButtons(container, variant) {
            if (!container) return;

            const loginButtonClasses = variant === 'desktop'
                ? 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform'
                : 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform w-full text-center mb-2';

            // Use clean URL for localhost, full redirect for production
            const loginUrl = this.config.isDevelopment
                ? 'https://accounts.classifast.com/sign-in'
                : `https://accounts.classifast.com/sign-in?redirect_url=${encodeURIComponent(window.location.href)}`;

            container.innerHTML = `
                <div class="flex ${variant === 'desktop' ? 'items-center' : 'flex-col'}">
                    <a href="${loginUrl}" 
                       class="${loginButtonClasses}">
                        Sign In
                    </a>
                </div>
            `;
            console.log('Rendered fallback login button (Clerk unavailable)');
        },

        async initializeClerkAuth() {
            if (this.isInitialized) {
                console.log('Clerk auth already initialized');
                return;
            }

            const desktopAuthContainer = document.getElementById('desktop-auth-container');
            const mobileAuthContainer = document.getElementById('mobile-auth-container');

            if (!desktopAuthContainer && !mobileAuthContainer) {
                console.log('Auth containers not found, retrying...');
                return;
            }

            try {
                // Check if Clerk is available
                if (typeof Clerk === 'undefined') {
                    throw new Error('Clerk is not loaded');
                }

                console.log('Loading Clerk...');
                await Clerk.load();

                // Check current auth state using up-to-date API
                const isSignedIn = !!Clerk.user; // Updated from deprecated isSignedIn
                console.log('Clerk loaded successfully, user authenticated:', isSignedIn);

                if (isSignedIn) {
                    // User is signed in - show avatar
                    this.renderClerkUserButton(desktopAuthContainer, 'desktop');
                    this.renderClerkUserButton(mobileAuthContainer, 'mobile');

                    // Add auth state change listener
                    if (Clerk.addListener) {
                        Clerk.addListener((event) => {
                            if (event.user === null) {
                                console.log('User signed out, refreshing auth UI');
                                this.renderLoggedOutButtons(desktopAuthContainer, 'desktop');
                                this.renderLoggedOutButtons(mobileAuthContainer, 'mobile');
                                this.bindAuthActions('desktop-login-button');
                                this.bindAuthActions('mobile-login-button');
                            }
                        });
                    }
                } else {
                    // User not signed in - show login button
                    this.renderLoggedOutButtons(desktopAuthContainer, 'desktop');
                    this.renderLoggedOutButtons(mobileAuthContainer, 'mobile');
                    this.bindAuthActions('desktop-login-button');
                    this.bindAuthActions('mobile-login-button');

                    // Add auth state change listener
                    if (Clerk.addListener) {
                        Clerk.addListener((event) => {
                            if (event.user !== null) {
                                console.log('User signed in, refreshing auth UI');
                                this.renderClerkUserButton(desktopAuthContainer, 'desktop');
                                this.renderClerkUserButton(mobileAuthContainer, 'mobile');
                            }
                        });
                    }
                }

                this.isInitialized = true;
                console.log('Clerk auth initialization complete');

            } catch (error) {
                console.error('Error initializing Clerk:', error);
                console.log('Using fallback authentication UI');

                // Fallback to account portal links
                this.renderFallbackButtons(desktopAuthContainer, 'desktop');
                this.renderFallbackButtons(mobileAuthContainer, 'mobile');

                this.isInitialized = true;
            }
        },

        // Check if Clerk script is loaded and available
        waitForClerk() {
            return new Promise((resolve, reject) => {
                const startTime = Date.now();

                const check = () => {
                    if (typeof Clerk !== 'undefined') {
                        resolve();
                    } else if (Date.now() - startTime > this.config.clerkLoadTimeout) {
                        reject(new Error('Clerk load timeout'));
                    } else {
                        this.clerkLoadRetries++;
                        if (this.clerkLoadRetries < this.config.maxRetries) {
                            setTimeout(check, this.config.retryInterval);
                        } else {
                            reject(new Error('Max retries exceeded'));
                        }
                    }
                };

                check();
            });
        },

        init() {
            console.log('Initializing Clerk Auth module...');

            // Setup modal handling
            this.setupClerkModalHandling();

            // Start initialization when DOM is ready
            const startInit = async () => {
                try {
                    // Wait for Clerk to be available
                    await this.waitForClerk();
                    // Initialize auth
                    await this.initializeClerkAuth();
                } catch (error) {
                    console.log('Clerk not available, using fallback:', error.message);
                    // Initialize with fallback directly
                    await this.initializeClerkAuth();
                }
            };

            // Check if DOM is already loaded
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', startInit);
            } else {
                startInit();
            }

            // Also handle dynamic content loading (for SPA/PWA)
            if (typeof window !== 'undefined') {
                window.addEventListener('load', () => {
                    if (!this.isInitialized) {
                        console.log('Window loaded but auth not initialized, forcing initialization');
                        this.initializeClerkAuth();
                    }
                });
            }
        }
    };

    // Initialize immediately
    window.ClerkAuth.init();

})();
