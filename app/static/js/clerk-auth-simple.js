// Simple Clerk Authentication using official SDK patterns
(function () {
    'use strict';

    async function init() {
        console.log('🚀 Initializing Clerk auth...');

        if (window.Clerk) {
            console.log('✅ Clerk object already present');
            await startClerk();
            return;
        }

        if (document.readyState === 'complete') {
            console.warn('⚠️ Page already loaded but Clerk not found');
            renderFallbackAuth();
            return;
        }

        console.log('⏳ Waiting for Clerk to load...');
        window.addEventListener('load', async () => {
            if (window.Clerk) {
                await startClerk();
            } else {
                console.error('❌ Clerk script failed to load');
                renderFallbackAuth();
            }
        });
    }

    async function startClerk() {
        try {
            await window.Clerk.load();
            console.log('✅ Clerk loaded successfully');

            // Initial render
            updateAuthUI();

            // Listen for auth state changes
            if (window.Clerk.addListener) {
                window.Clerk.addListener((payload) => {
                    console.log('🔄 Auth state changed');
                    updateAuthUI();
                });
            }
        } catch (err) {
            console.error('❌ Error initializing Clerk:', err);
            renderFallbackAuth();
        }
    }

    function updateAuthUI() {
        const user = window.Clerk.user;
        const desktopContainer = document.getElementById('desktop-auth-container');
        const mobileContainer = document.getElementById('mobile-auth-container');

        // Clear containers
        if (desktopContainer) desktopContainer.innerHTML = '';
        if (mobileContainer) mobileContainer.innerHTML = '';

        if (user) {
            console.log('👤 User is signed in');
            // Render User Button
            mountUserButton(desktopContainer, 'desktop');
            mountUserButton(mobileContainer, 'mobile');
        } else {
            console.log('👤 User is signed out');
            // Render Sign In Button
            renderSignInButton(desktopContainer, 'desktop');
            renderSignInButton(mobileContainer, 'mobile');

            // Try to open Google One Tap
            openGoogleOneTap();
        }
    }

    function openGoogleOneTap() {
        try {
            if (window.Clerk && window.Clerk.openGoogleOneTap) {
                const params = {
                    cancelOnTapOutside: false,
                    itpSupport: true,
                    fedCmSupport: true
                };
                window.Clerk.openGoogleOneTap(params);
                console.log('✅ Google One Tap opened');
            }
        } catch (err) {
            console.error('❌ Error opening Google One Tap:', err);
        }
    }

    function mountUserButton(container, type) {
        if (!container) return;

        const el = document.createElement('div');
        // Add an ID for easier debugging if needed, though not strictly required by Clerk
        el.id = `clerk-user-button-${type}`;
        container.appendChild(el);

        try {
            window.Clerk.mountUserButton(el, {
                appearance: {
                    elements: {
                        userButtonAvatarBox: 'w-8 h-8',
                        userButtonBox: 'h-8'
                    }
                }
            });
        } catch (err) {
            console.error(`❌ Error mounting ${type} user button:`, err);
        }
    }

    function renderSignInButton(container, type) {
        if (!container) return;

        const button = document.createElement('div');

        // Apply styles based on type (preserving original styles)
        if (type === 'desktop') {
            button.id = 'clerk-sign-in-button-desktop';
            button.className = 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-6 py-1 rounded text-base transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded';
        } else {
            button.id = 'clerk-sign-in-button-mobile';
            button.className = 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-6 py-1 rounded text-base transition-all duration-150 ease-in-out transform cursor-pointer w-full text-center mb-2 auth-loaded';
        }

        button.textContent = 'Sign In';

        button.addEventListener('click', (e) => {
            e.preventDefault();
            if (window.Clerk && window.Clerk.openSignIn) {
                window.Clerk.openSignIn();
            } else {
                // Fallback redirect
                window.location.href = 'https://accounts.classifast.com/sign-in?redirect_url=' + encodeURIComponent(window.location.href);
            }
        });

        container.appendChild(button);
    }

    function renderFallbackAuth() {
        console.log('🚨 Rendering fallback auth UI');
        const desktopContainer = document.getElementById('desktop-auth-container');
        const mobileContainer = document.getElementById('mobile-auth-container');

        const fallbackUrl = 'https://accounts.classifast.com/sign-in?redirect_url=' + encodeURIComponent(window.location.href);
        const className = 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white font-semibold px-6 py-1 rounded text-base transition-all duration-150 ease-in-out transform auth-loaded';

        const createFallbackLink = () => {
            const a = document.createElement('a');
            a.href = fallbackUrl;
            a.className = className;
            a.textContent = 'Sign In';
            return a;
        };

        if (desktopContainer) {
            desktopContainer.innerHTML = '';
            desktopContainer.appendChild(createFallbackLink());
        }
        if (mobileContainer) {
            mobileContainer.innerHTML = '';
            const link = createFallbackLink();
            // Add mobile specific classes if needed, usually the generic one is fine or we add w-full
            // The original fallback used the same classes for both, so we stick to that or slight adjustments.
            // Original mobile fallback: same classes.
            mobileContainer.appendChild(link);
        }
    }

    // Start the initialization
    init();

})();
