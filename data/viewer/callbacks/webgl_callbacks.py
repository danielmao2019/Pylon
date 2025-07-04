"""WebGL-specific callbacks for point cloud visualization."""
from typing import List, Dict, Any
from dash import Input, Output, State, clientside_callback, ALL
from data.viewer.callbacks.registry import callback


def register_webgl_callbacks(app):
    """Register all WebGL-related callbacks with the Dash app."""
    
    # Register pattern-based clientside callback for WebGL initialization
    app.clientside_callback(
        """
        function(trigger_values, data_values) {
            console.log('=== WebGL Pattern Callback Started ===');
            console.log('Trigger values:', trigger_values);
            console.log('Data values length:', data_values ? data_values.length : 'undefined');
            
            let results = [];
            
            // Process each trigger/data pair
            if (trigger_values && data_values) {
                for (let i = 0; i < trigger_values.length; i++) {
                    if (trigger_values[i] && data_values[i]) {
                        try {
                            console.log('Processing WebGL component', i);
                            
                            // Parse config
                            const config = JSON.parse(data_values[i]);
                            console.log('Config loaded:', config.pointCloudData.point_count, 'points');
                            
                            // Find the container element - Dash converts pattern IDs to strings
                            // Pattern: {"type":"webgl-container","index":"webgl-point-cloud-abc123"}
                            // becomes: webgl-container-webgl-point-cloud-abc123
                            const containers = document.querySelectorAll('[id*="webgl-container"]');
                            let container = null;
                            
                            // Find the matching container by index
                            if (containers.length > i) {
                                container = containers[i];
                            }
                            
                            if (!container) {
                                console.error('Container not found for index', i);
                                results.push('Error: Container not found');
                                continue;
                            }
                            
                            console.log('Found container:', container.id);
                            
                            // Create canvas with simple point cloud visualization
                            const canvas = document.createElement('canvas');
                            canvas.width = container.clientWidth || 500;
                            canvas.height = container.clientHeight || 500;
                            canvas.style.width = '100%';
                            canvas.style.height = '100%';
                            canvas.style.backgroundColor = '#87CEEB';  // Sky blue
                            
                            container.innerHTML = '';
                            container.appendChild(canvas);
                            
                            // Draw points on canvas
                            const ctx = canvas.getContext('2d');
                            if (ctx) {
                                // Draw title
                                ctx.fillStyle = '#333';
                                ctx.font = '16px Arial';
                                ctx.textAlign = 'center';
                                ctx.fillText('WebGL Point Cloud (2D Fallback)', canvas.width/2, 30);
                                
                                // Draw point count
                                ctx.font = '14px Arial';
                                ctx.fillText(config.pointCloudData.point_count + ' points loaded', canvas.width/2, 60);
                                
                                // Draw some sample points
                                const positions = config.pointCloudData.positions;
                                const colors = config.pointCloudData.colors;
                                const pointSize = config.pointSize || 2;
                                
                                // Map 3D coordinates to 2D canvas
                                const centerX = canvas.width / 2;
                                const centerY = canvas.height / 2;
                                const scale = Math.min(canvas.width, canvas.height) / 4;
                                
                                // Draw sample of points (max 1000 for performance)
                                const maxPoints = Math.min(1000, positions.length / 3);
                                const step = Math.max(1, Math.floor(positions.length / 3 / maxPoints));
                                
                                for (let j = 0; j < positions.length; j += 3 * step) {
                                    const x = centerX + positions[j] * scale;
                                    const y = centerY - positions[j + 1] * scale;  // Flip Y for canvas
                                    
                                    // Use point color
                                    const r = Math.floor(colors[j] * 255);
                                    const g = Math.floor(colors[j + 1] * 255);
                                    const b = Math.floor(colors[j + 2] * 255);
                                    
                                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                                    ctx.beginPath();
                                    ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
                                    ctx.fill();
                                }
                                
                                console.log('Canvas point cloud drawn');
                                results.push('WebGL callback completed - ' + config.pointCloudData.point_count + ' points visualized');
                            } else {
                                console.error('Could not get canvas context');
                                results.push('Error: Could not get canvas context');
                            }
                            
                        } catch (error) {
                            console.error('WebGL callback error:', error);
                            results.push('Error: ' + error.message);
                        }
                    } else {
                        results.push('No trigger/data');
                    }
                }
            }
            
            console.log('WebGL pattern callback results:', results);
            return results;
        }
        """,
        Output({'type': 'webgl-status', 'index': ALL}, 'children'),
        Input({'type': 'webgl-trigger', 'index': ALL}, 'children'),
        State({'type': 'webgl-data', 'index': ALL}, 'children')
    )