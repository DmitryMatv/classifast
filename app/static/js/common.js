// Shared JavaScript functionality for Classifast application

// Mobile menu functionality
class MobileMenu {
    constructor() {
        this.button = null;
        this.menu = null;
        this.hamburger = null;
        this.init();
    }

    init() {
        this.button = document.getElementById('mobile-menu-button');
        this.menu = document.getElementById('mobile-menu');
        this.hamburger = document.querySelector('.hamburger');

        if (!this.button || !this.menu || !this.hamburger) return;

        this.button.addEventListener('click', () => this.toggle());

        // Close on link click
        const links = this.menu.querySelectorAll('a');
        links.forEach(link => {
            link.addEventListener('click', () => this.close());
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!this.menu.contains(e.target) && !this.button.contains(e.target)) {
                this.close();
            }
        });

        // Close on ESC key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.menu.classList.contains('active')) {
                this.close();
                this.button.focus();
            }
        });
    }

    toggle() {
        const isActive = this.menu.classList.toggle('active');
        this.hamburger.classList.toggle('active');
        this.button.setAttribute('aria-expanded', isActive);
    }

    close() {
        this.menu.classList.remove('active');
        this.hamburger.classList.remove('active');
        this.button.setAttribute('aria-expanded', 'false');
    }
}

// Copy URL functionality
class ShareLink {
    static async copyShareableLink() {
        const url = window.location.href;
        const button = document.getElementById('share-button');

        try {
            await navigator.clipboard.writeText(url);
            this.showFeedback(button);
        } catch (err) {
            console.error('Could not copy URL: ', err);
            this.fallbackCopy(url, button);
        }
    }

    static showFeedback(button) {
        if (!button) return;

        const originalText = button.innerHTML;
        button.innerHTML = 'Copied!';
        button.classList.add('bg-green-600', 'hover:bg-green-700');

        setTimeout(() => {
            button.innerHTML = originalText;
            button.classList.remove('bg-green-600', 'hover:bg-green-700');
        }, 2000);
    }

    static fallbackCopy(url, button) {
        const textArea = document.createElement('textarea');
        textArea.value = url;
        document.body.appendChild(textArea);
        textArea.select();

        try {
            document.execCommand('copy');
            console.log('URL copied using fallback');
            this.showFeedback(button);
        } catch (fallbackErr) {
            console.error('Fallback copy failed: ', fallbackErr);
        }

        document.body.removeChild(textArea);
    }
}

// Textarea enhanced functionality
class TextareaEnhancer {
    constructor(textareaId) {
        this.textarea = document.getElementById(textareaId);
        if (this.textarea) {
            this.init();
        }
    }

    init() {
        this.textarea.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.submitForm();
            }
        });
    }

    submitForm() {
        const form = this.textarea.closest('form');
        if (form) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.classList.add('active', 'scale-95');
                setTimeout(() => {
                    submitBtn.classList.remove('active', 'scale-95');
                }, 150);
                submitBtn.click();
            } else {
                form.submit();
            }
        }
    }

    getValue() {
        return this.textarea?.value || '';
    }

    setValue(value) {
        if (this.textarea) {
            this.textarea.value = value;
        }
    }
}

// Toggle functionality for classifier description sections
class DescriptionToggle {
    constructor() {
        this.init();
    }

    init() {
        const toggle = document.getElementById('description-toggle');
        const content = document.getElementById('description-content');
        const container = document.getElementById('description-container');

        if (!toggle || !content || !container) return;

        // Hide entire block if description empty
        const text = content.textContent || content.innerText || '';
        if (!text.trim()) {
            toggle.style.display = 'none';
            container.style.display = 'none';
            return;
        }

        this.setupToggle(toggle, content);
    }

    setupToggle(toggle, content) {
        const logos = document.querySelectorAll('[data-classifier-logo]');
        const learnLabel = toggle.getAttribute('aria-label')?.replace(' button', '') || 'Learn more';
        const showLessLabel = 'Show less';

        // Initialize state: hidden
        content.style.display = 'none';
        content.setAttribute('aria-hidden', 'true');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.textContent = learnLabel;

        toggle.addEventListener('click', (e) => {
            e.preventDefault();
            const isHidden = content.style.display === 'none' || content.style.display === '';

            if (isHidden) {
                this.showContent(content, toggle, logos, showLessLabel);
            } else {
                this.hideContent(content, toggle, logos, learnLabel);
            }
        });
    }

    showContent(content, toggle, logos, showLessLabel) {
        content.style.display = 'block';
        content.setAttribute('aria-hidden', 'false');
        toggle.setAttribute('aria-expanded', 'true');
        toggle.textContent = showLessLabel;

        logos.forEach(logo => {
            if (!logo.dataset.originalDisplay) {
                logo.dataset.originalDisplay = logo.style.display || '';
            }
            logo.style.display = 'none';
        });
    }

    hideContent(content, toggle, logos, learnLabel) {
        content.style.display = 'none';
        content.setAttribute('aria-hidden', 'true');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.textContent = learnLabel;

        logos.forEach(logo => {
            const original = logo.dataset.originalDisplay || '';
            logo.style.display = original;
        });
    }
}

// Simple Clerk Authentication using official SDK patterns
class ClerkAuth {
    constructor() {
        this.init();
    }

    async init() {
        // console.log('🚀 Initializing Clerk auth...');

        if (window.Clerk) {
            // console.log('✅ Clerk object already present');
            await this.startClerk();
            return;
        }

        if (document.readyState === 'complete') {
            // console.warn('⚠️ Page already loaded but Clerk not found');
            this.renderFallbackAuth();
            return;
        }

        // console.log('⏳ Waiting for Clerk to load...');
        window.addEventListener('load', async () => {
            if (window.Clerk) {
                await this.startClerk();
            } else {
                console.error('❌ Clerk script failed to load');
                this.renderFallbackAuth();
            }
        });
    }

    async startClerk() {
        try {
            await window.Clerk.load();
            // console.log('✅ Clerk loaded successfully');

            // Initial render
            this.updateAuthUI();

            // Listen for auth state changes
            if (window.Clerk.addListener) {
                window.Clerk.addListener((payload) => {
                    // console.log('🔄 Auth state changed');
                    this.updateAuthUI();
                });
            }
        } catch (err) {
            console.error('❌ Error initializing Clerk:', err);
            this.renderFallbackAuth();
        }
    }

    updateAuthUI() {
        const user = window.Clerk.user;
        const desktopContainer = document.getElementById('desktop-auth-container');
        const mobileContainer = document.getElementById('mobile-auth-container');

        // Clear containers
        if (desktopContainer) desktopContainer.innerHTML = '';
        if (mobileContainer) mobileContainer.innerHTML = '';

        if (user) {
            // console.log('👤 User is signed in');

            // Render Upgrade Button
            this.renderUpgradeButton(desktopContainer, 'desktop');
            this.renderUpgradeButton(mobileContainer, 'mobile');

            // Render User Button
            this.mountUserButton(desktopContainer, 'desktop');
            this.mountUserButton(mobileContainer, 'mobile');
        } else {
            // console.log('👤 User is signed out');
            // Render Sign In Button
            this.renderSignInButton(desktopContainer, 'desktop');
            this.renderSignInButton(mobileContainer, 'mobile');

            // Try to open Google One Tap
            this.openGoogleOneTap();
        }
    }

    renderUpgradeButton(container, type) {
        if (!container) return;

        // Check if user is already pro
        const user = window.Clerk.user;
        const isPro = user.publicMetadata?.tier === 'pro';
        if (isPro) return;

        const button = document.createElement('button');
        button.textContent = 'Upgrade to Pro';

        if (type === 'desktop') {
            button.className = 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white px-3 py-1 rounded mr-4 text-sm font-medium transition-all shadow-sm';
        } else {
            button.className = 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white px-4 py-2 rounded w-full mb-4 text-base transition-all shadow-sm';
        }

        button.onclick = async () => {
            try {
                const originalText = button.textContent;
                button.disabled = true;
                button.textContent = 'Preparing...';

                const token = await window.Clerk.session.getToken();
                const response = await fetch('/api/create-checkout', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    // TODO: Replace with actual Polar Product ID
                    body: JSON.stringify({
                        product_id: 'b71bb3cf-a34f-428f-bad8-142937c10c8f'
                    })
                });

                if (!response.ok) throw new Error('Checkout creation failed');

                const data = await response.json();
                if (data.url) {
                    window.location.href = data.url;
                } else {
                    throw new Error('No checkout URL returned');
                }
            } catch (err) {
                console.error('Upgrade failed:', err);
                button.textContent = 'Error';
                setTimeout(() => {
                    button.disabled = false;
                    button.textContent = 'Upgrade to Pro';
                }, 3000);
            }
        };

        container.appendChild(button);
    }

    openGoogleOneTap() {
        try {
            if (window.Clerk && window.Clerk.openGoogleOneTap) {
                const params = {
                    cancelOnTapOutside: false,
                    itpSupport: true,
                    fedCmSupport: true
                };
                window.Clerk.openGoogleOneTap(params);
                // console.log('✅ Google One Tap opened');
            }
        } catch (err) {
            console.error('❌ Error opening Google One Tap:', err);
        }
    }

    mountUserButton(container, type) {
        if (!container) return;

        const el = document.createElement('div');
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

    renderSignInButton(container, type) {
        if (!container) return;

        const button = document.createElement('div');

        // Apply styles based on type
        if (type === 'desktop') {
            button.id = 'clerk-sign-in-button-desktop';
            button.className = 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform cursor-pointer auth-loaded';
        } else {
            button.id = 'clerk-sign-in-button-mobile';
            button.className = 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform cursor-pointer w-full text-center mb-2 auth-loaded';
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

    renderFallbackAuth() {
        // console.log('🚨 Rendering fallback auth UI');
        const desktopContainer = document.getElementById('desktop-auth-container');
        const mobileContainer = document.getElementById('mobile-auth-container');

        const fallbackUrl = 'https://accounts.classifast.com/sign-in?redirect_url=' + encodeURIComponent(window.location.href);
        const className = 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 active:scale-95 text-white px-4 py-1 rounded text-base transition-all duration-150 ease-in-out transform auth-loaded';

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
            mobileContainer.appendChild(link);
        }
    }
}

// Result copy functionality with tooltip
class ResultCopier {
    constructor() {
        this.init();
    }

    init() {
        // Expose global function for inline HTML onclick handlers
        window.copyOriginalId = (text, buttonElement) => this.copy(text, buttonElement);
    }

    copy(text, buttonElement) {
        if (!navigator.clipboard) {
            this.fallbackCopy(text, buttonElement);
            return;
        }

        navigator.clipboard.writeText(text).then(() => {
            this.showTooltip(buttonElement, 'Copied!');
        }).catch((err) => {
            console.error('Async: Could not copy text: ', err);
            this.showTooltip(buttonElement, 'Copy failed');
        });
    }

    fallbackCopy(text, buttonElement) {
        // Fallback for older browsers or insecure contexts (e.g. http)
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed"; // Prevent scrolling to bottom
        textArea.style.opacity = "0"; // Hide the textarea
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            document.execCommand('copy');
            this.showTooltip(buttonElement, 'Copied!');
        } catch (err) {
            console.error('Fallback: Oops, unable to copy', err);
            this.showTooltip(buttonElement, 'Copy failed');
        }
        document.body.removeChild(textArea);
    }

    showTooltip(buttonElement, message) {
        const tooltip = document.createElement('span');
        tooltip.textContent = message;
        // Basic styling for the tooltip
        tooltip.style.position = 'absolute';
        tooltip.style.backgroundColor = 'black';
        tooltip.style.color = 'white';
        tooltip.style.padding = '4px 8px';
        tooltip.style.borderRadius = '4px';
        tooltip.style.fontSize = '1.125rem'; // text-base (medium)
        tooltip.style.zIndex = '1000';     // Ensure it's on top
        tooltip.style.textAlign = 'center';

        // Append to body to avoid clipping issues and for correct initial dimension calculation
        document.body.appendChild(tooltip);

        const buttonRect = buttonElement.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect(); // Get dimensions after appending and styling

        // Position above the button, centered, with scroll offset
        let top = buttonRect.top + window.scrollY - tooltipRect.height - 5; // 5px spacing
        let left = buttonRect.left + window.scrollX + (buttonRect.width / 2) - (tooltipRect.width / 2);

        // Adjust if tooltip goes off-screen (viewport relative checks)
        if (buttonRect.top - tooltipRect.height - 5 < 0) { // Not enough space above
            top = buttonRect.bottom + window.scrollY + 5; // Position below
        }
        if (left - window.scrollX < 0) { // Off-screen left
            left = window.scrollX;
        }
        if (left - window.scrollX + tooltipRect.width > window.innerWidth) { // Off-screen right
            left = window.scrollX + window.innerWidth - tooltipRect.width;
        }

        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;

        buttonElement.disabled = true; // Disable button

        setTimeout(function () {
            if (tooltip.parentNode) {
                tooltip.parentNode.removeChild(tooltip);
            }
            buttonElement.disabled = false; // Re-enable button
        }, 500); // Tooltip lasts for 500 milliseconds
    }
}

// Initialize common functionality
document.addEventListener('DOMContentLoaded', function () {
    new MobileMenu();
    new DescriptionToggle();
    new ClerkAuth();
    new TextareaEnhancer('product_description_area');
    new ResultCopier();
});
