/**
 * WebGL Point Cloud Initialization Helper
 * Minimal wrapper to initialize point clouds with configuration
 */

function initWebGLPointCloudWithConfig(config) {
    const { containerId, pointCloudData, pointSize, pointOpacity, cameraState } = config;

    function tryInit() {
        // Check if all dependencies are loaded
        if (typeof THREE !== 'undefined' &&
            typeof THREE.OrbitControls !== 'undefined' &&
            typeof initWebGLPointCloud === 'function') {

            const container = document.getElementById(containerId);
            if (container) {
                initWebGLPointCloud(
                    container,
                    pointCloudData,
                    pointSize,
                    pointOpacity,
                    cameraState
                );
            } else {
                console.error('Container not found:', containerId);
            }
        } else {
            // Retry after a short delay
            setTimeout(tryInit, 100);
        }
    }

    // Start initialization
    tryInit();
}

// Make available globally
if (typeof window !== 'undefined') {
    window.initWebGLPointCloudWithConfig = initWebGLPointCloudWithConfig;
}