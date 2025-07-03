/**
 * WebGL Point Cloud Renderer using Three.js
 * High-performance point cloud visualization for large datasets
 */

// Global Three.js imports (assumes Three.js is loaded)
const { Scene, PerspectiveCamera, WebGLRenderer, BufferGeometry, BufferAttribute, 
        PointsMaterial, Points, Color, Vector3, Box3 } = THREE;

// OrbitControls for camera interaction
const { OrbitControls } = THREE;

/**
 * Initialize WebGL point cloud renderer with Three.js
 * @param {HTMLElement} container - DOM container element
 * @param {Object} data - Point cloud data object
 * @param {number} pointSize - Size of points
 * @param {number} opacity - Point opacity
 * @param {Object} cameraState - Initial camera state
 */
function initWebGLPointCloud(container, data, pointSize, opacity, cameraState) {
    // Check WebGL support
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    
    if (!gl) {
        // Fallback for no WebGL support
        container.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #856404; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
                <h5>WebGL Not Supported</h5>
                <p>Your browser doesn't support WebGL. Please use a modern browser or enable WebGL.</p>
                <p>Point cloud data: ${data.point_count.toLocaleString()} points</p>
            </div>
        `;
        return;
    }

    try {
        // Create Three.js scene
        const scene = new Scene();
        scene.background = new Color(0xffffff);

        // Set up camera
        const aspect = container.clientWidth / container.clientHeight;
        const camera = new PerspectiveCamera(75, aspect, 0.1, data.bbox_size * 10);
        
        // Position camera based on provided state or defaults
        const cameraPos = cameraState.position || [data.bbox_size * 2, data.bbox_size * 2, data.bbox_size * 2];
        camera.position.set(cameraPos[0], cameraPos[1], cameraPos[2]);
        
        // Set camera target
        const target = cameraState.target || data.bbox_center;
        camera.lookAt(new Vector3(target[0], target[1], target[2]));

        // Create WebGL renderer
        const renderer = new WebGLRenderer({ 
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true
        });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit for performance
        
        // Clear container and add canvas
        container.innerHTML = '';
        container.appendChild(renderer.domElement);

        // Create point cloud geometry
        const geometry = new BufferGeometry();
        
        // Set positions (convert flat array to Float32Array)
        const positions = new Float32Array(data.positions);
        geometry.setAttribute('position', new BufferAttribute(positions, 3));
        
        // Set colors (convert flat array to Float32Array)
        const colors = new Float32Array(data.colors);
        geometry.setAttribute('color', new BufferAttribute(colors, 3));

        // Create material
        const material = new PointsMaterial({
            size: pointSize,
            opacity: opacity,
            transparent: opacity < 1.0,
            vertexColors: true,
            sizeAttenuation: true
        });

        // Create points object
        const points = new Points(geometry, material);
        scene.add(points);

        // Set up orbit controls for interaction
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(target[0], target[1], target[2]);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.enablePan = true;
        controls.enableZoom = true;
        controls.enableRotate = true;
        
        // Control limits
        controls.minDistance = data.bbox_size * 0.1;
        controls.maxDistance = data.bbox_size * 5;
        
        // Performance optimizations
        controls.enableKeys = false; // Disable keyboard controls for performance
        controls.screenSpacePanning = false;

        // Animation loop
        let animationId;
        function animate() {
            animationId = requestAnimationFrame(animate);
            
            // Update controls
            controls.update();
            
            // Render scene
            renderer.render(scene, camera);
        }

        // Handle window resize
        const resizeObserver = new ResizeObserver(entries => {
            const entry = entries[0];
            const { width, height } = entry.contentRect;
            
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        });
        resizeObserver.observe(container);

        // Keyboard shortcuts
        const keyHandler = (event) => {
            switch(event.key.toLowerCase()) {
                case 'r':
                    // Reset view
                    camera.position.set(cameraPos[0], cameraPos[1], cameraPos[2]);
                    controls.target.set(target[0], target[1], target[2]);
                    controls.update();
                    break;
                case '+':
                case '=':
                    // Increase point size
                    material.size = Math.min(material.size + 0.5, 20);
                    break;
                case '-':
                    // Decrease point size
                    material.size = Math.max(material.size - 0.5, 0.5);
                    break;
            }
        };
        
        // Add event listeners
        document.addEventListener('keydown', keyHandler);

        // Start animation
        animate();

        // Performance monitoring
        let frameCount = 0;
        let lastTime = performance.now();
        const fpsDisplay = document.createElement('div');
        fpsDisplay.style.cssText = `
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 5px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 12px;
            z-index: 1000;
        `;
        container.appendChild(fpsDisplay);
        container.style.position = 'relative';

        function updateFPS() {
            frameCount++;
            const currentTime = performance.now();
            if (currentTime - lastTime >= 1000) {
                const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                fpsDisplay.textContent = `${fps} FPS | ${data.point_count.toLocaleString()} pts`;
                frameCount = 0;
                lastTime = currentTime;
            }
            requestAnimationFrame(updateFPS);
        }
        updateFPS();

        // Cleanup function for component unmounting
        const cleanup = () => {
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
            document.removeEventListener('keydown', keyHandler);
            resizeObserver.disconnect();
            
            // Dispose Three.js resources
            geometry.dispose();
            material.dispose();
            renderer.dispose();
            
            // Clear container
            container.innerHTML = '';
        };

        // Store cleanup function on container for external access
        container._webglCleanup = cleanup;

    } catch (error) {
        // Error fallback
        container.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px;">
                <h5>WebGL Initialization Error</h5>
                <p>Error: ${error.message}</p>
                <p>Point count: ${data.point_count.toLocaleString()}</p>
                <details style="margin-top: 10px;">
                    <summary>Error Details</summary>
                    <pre style="text-align: left; font-size: 11px; margin-top: 5px;">${error.stack}</pre>
                </details>
            </div>
        `;
        console.error('WebGL Point Cloud Error:', error);
    }
}

// Make functions available globally
if (typeof window !== 'undefined') {
    window.initWebGLPointCloud = initWebGLPointCloud;
}