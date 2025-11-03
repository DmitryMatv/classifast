// Shared Clerk Authentication Module
(function () {
    'use strict';

    // Clerk Authentication Module
    window.ClerkAuth = {

        setupClerkModalHandling() {
            // More precise modal detection - only hide scrollbar for actual modal overlays
            const checkForClerkModal = () => {
                // Look specifically for modal/overlay elements, not all Clerk components
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

            // Watch for DOM changes that might indicate modal state changes
            if ('MutationObserver' in window) {
                const observer = new MutationObserver(() => {
                    checkForClerkModal();
                });

                observer.observe(document.body, {
                    attributes: true,
                    attributeFilter: ['style'],
                    subtree: true,
                    childList: true
                });
            }

            // Also check when Clerk buttons are clicked
            document.addEventListener('click', (e) => {
                if (e.target.id?.includes('sign-in') || e.target.id?.includes('sign-up')) {
                    setTimeout(checkForClerkModal, 100);
                }
            });

            // Initial check
            setTimeout(checkForClerkModal, 100);
        },

        renderLoggedOutButtons(container, variant) {
            if (!container) return;
            if (variant === 'desktop') {
                container.innerHTML = `
                    <div class="flex space-x-6 items-center">
                        <button type="button" id="desktop-sign-in-button" class="nav-auth-link">
                            Sign In
                        </button>
                        <button type="button" id="desktop-sign-up-button" class="nav-auth-link">
                            Sign Up
                        </button>
                    </div>
                `;
            } else {
                container.innerHTML = `
                    <div class="flex flex-col">
                        <button type="button" id="mobile-sign-in-button" class="mobile-nav-button">
                            Sign In
                        </button>
                        <button type="button" id="mobile-sign-up-button" class="mobile-nav-button">
                            Sign Up
                        </button>
                    </div>
                `;
            }
        },

        bindAuthActions(signInId, signUpId) {
            const signInButton = document.getElementById(signInId);
            if (signInButton) {
                signInButton.addEventListener('click', () => {
                    Clerk.openSignIn({
                        afterSignInUrl: window.location.href,
                        afterSignUpUrl: window.location.href
                    });
                });
            }

            const signUpButton = document.getElementById(signUpId);
            if (signUpButton) {
                signUpButton.addEventListener('click', () => {
                    Clerk.openSignUp({
                        afterSignInUrl: window.location.href,
                        afterSignUpUrl: window.location.href
                    });
                });
            }
        },

        async initClerkAuth() {
            const desktopAuthContainer = document.getElementById('desktop-auth-container');
            const mobileAuthContainer = document.getElementById('mobile-auth-container');

            try {
                await Clerk.load();
                console.log('Clerk loaded successfully, isSignedIn:', Clerk.isSignedIn);

                if (Clerk.isSignedIn) {
                    if (desktopAuthContainer) {
                        desktopAuthContainer.innerHTML = '<div id="desktop-user-button"></div>';
                        const desktopUserButtonDiv = document.getElementById('desktop-user-button');
                        Clerk.mountUserButton(desktopUserButtonDiv);
                    }

                    if (mobileAuthContainer) {
                        mobileAuthContainer.innerHTML = '<div id="mobile-user-button"></div>';
                        const mobileUserButtonDiv = document.getElementById('mobile-user-button');
                        Clerk.mountUserButton(mobileUserButtonDiv);
                    }
                } else {
                    this.renderLoggedOutButtons(desktopAuthContainer, 'desktop');
                    this.renderLoggedOutButtons(mobileAuthContainer, 'mobile');
                    this.bindAuthActions('desktop-sign-in-button', 'desktop-sign-up-button');
                    this.bindAuthActions('mobile-sign-in-button', 'mobile-sign-up-button');
                }
            } catch (error) {
                console.error('Error initializing Clerk:', error);
                [desktopAuthContainer, mobileAuthContainer].forEach(container => {
                    if (!container) return;
                    container.innerHTML = `
                        <a href="https://darling-seagull-34.clerk.accounts.dev/sign-in?redirect_url=${window.location.href}"
                           class="bg-sky-600 text-white px-4 py-2 rounded hover:bg-sky-700 transition-colors block text-center">
                            Sign In
                        </a>
                    `;
                });
            }
        },

        init() {
            // Setup modal handling first
            this.setupClerkModalHandling();

            // Try to initialize Clerk when it's available
            const tryInitClerk = () => {
                if (typeof Clerk !== 'undefined') {
                    this.initClerkAuth();
                } else {
                    setTimeout(tryInitClerk, 100);
                }
            };

            tryInitClerk();
        }
    };

    // Initialize Clerk when DOM is ready
    document.addEventListener('DOMContentLoaded', function () {
        window.ClerkAuth.init();
    });

})();
