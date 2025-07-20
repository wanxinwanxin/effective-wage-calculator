// Configuration for different environments
const CONFIG = {
    // Auto-detect environment
    getApiBase() {
        // If we're running on localhost, use local backend
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }
        
        // For GitHub Pages deployment, use deployed backend
        // Update this URL after deploying to Railway/Render
        if (window.location.hostname.includes('github.io')) {
            return 'https://effective-wage-calculator-production.up.railway.app'; // Replace with your deployed URL
        }
        
        // Default fallback
        return 'http://localhost:8000';
    },
    
    // Other configuration options
    DEFAULT_TIMEOUT: 30000,
    ENABLE_DEBUG: window.location.hostname === 'localhost',
    
    // Feature flags
    FEATURES: {
        AI_EXPLANATIONS: true,
        DETAILED_LOGGING: true,
        RESEARCH_CITATIONS: true
    }
};

// Make config globally available
window.EHW_CONFIG = CONFIG; 