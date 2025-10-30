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

// URL parameter handling utilities
class UrlUtils {
    // Extract URL parameters (simplified, hardcoded list for reliability)
    static getUrlParams() {
        const pathParts = window.location.pathname.split('/');
        // Hardcoded list of supported classifiers for reliability
        const supportedClassifiers = ['etim', 'unspsc', 'naics', 'isic', 'hs', 'hts', 'cn', 'nace', 'cpv', 'nsn'];
        const classifierIndex = pathParts.findIndex(part =>
            supportedClassifiers.includes(part)
        );

        if (classifierIndex !== -1 && pathParts.length > classifierIndex + 1) {
            const searchQuery = decodeURIComponent(
                pathParts.slice(classifierIndex + 1).join('/')
            ).replace(/-/g, ' ');
            return {
                search: searchQuery || '',
                version: new URLSearchParams(window.location.search).get('version') || ''
            };
        }

        return {
            search: '',
            version: new URLSearchParams(window.location.search).get('version') || ''
        };
    }

    // Slugify function for SEO-friendly URLs
    static slugify(text) {
        return text.toString().toLowerCase()
            .replace(/[^\w\s-]/g, '') // Remove special characters
            .replace(/[\s_-]+/g, '-') // Replace spaces and underscores with hyphens
            .replace(/^-+|-+$/g, ''); // Remove leading/trailing hyphens
    }

    // Update URL parameters
    static updateUrlParams(search, version) {
        const classifierType = window.location.pathname.split('/')[1];
        let newUrl;

        if (search && search.trim()) {
            const slug = this.slugify(search);
            newUrl = `/${classifierType}/${slug}`;
            if (version && version !== '') {
                newUrl += `?version=${encodeURIComponent(version)}`;
            }
        } else {
            newUrl = `/${classifierType}`;
            if (version && version !== '') {
                newUrl += `?version=${encodeURIComponent(version)}`;
            }
        }

        window.history.replaceState({ search, version }, '', newUrl);
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

// Initialize common functionality
document.addEventListener('DOMContentLoaded', function () {
    // Initialize mobile menu
    new MobileMenu();
});
