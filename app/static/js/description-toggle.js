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

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function () {
    new DescriptionToggle();
});
