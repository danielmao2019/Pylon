# Implementation Plan: Density Control for 3D Viewer

## Overview
Add a "Density" slider control to the 3D viewer that appears when LOD type is set to "none". This will allow users to control the percentage of points displayed (1% to 100%) with a caching system for performance optimization.

## Current System Analysis

### LOD System Structure
- **LOD Types**: `none`, `continuous`, `discrete` (configured in `ViewerSettings.LOD_TYPE_OPTIONS`)
- **LOD Processing**: Done in `apply_lod_to_point_cloud()` in `/data/viewer/utils/point_cloud.py`
- **Caching**: 
  - Discrete LOD uses global caches (`_global_lod_cache`, `_global_original_cache`) 
  - Continuous LOD computes dynamically (no caching)
  - No LOD (`none`) currently returns original points without processing

### Current Control Flow
1. UI controls in `/data/viewer/layout/controls/controls_3d.py`
2. Settings managed via callbacks in `/data/viewer/callbacks/three_d_settings.py` 
3. Settings stored in `3d-settings-store` and synced to backend
4. Display functions call `create_point_cloud_figure()` with LOD settings
5. `create_point_cloud_figure()` calls `apply_lod_to_point_cloud()` for processing

## Implementation Design

### 1. Density Caching System

#### Cache Structure
```python
# New global cache for density-based subsampling
_global_density_cache: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
# Key format: point_cloud_id -> {percentage: point_cloud_dict}
```

#### Cache Management
- **Cache Key**: `point_cloud_id` (same format as discrete LOD)
- **Cache Value**: Dictionary mapping density percentage (1-100) to subsampled point clouds
- **Invalidation**: Clear cache when point cloud data changes
- **Memory Management**: LRU-style eviction if needed

#### Implementation Location
Create new file: `/data/viewer/utils/density_lod.py`

### 2. Settings Integration

#### Settings Configuration (`/data/viewer/utils/settings_config.py`)
```python
# Add to DEFAULT_3D_SETTINGS
'density_percentage': 100  # Default 100% (all points)

# Add validation in validate_3d_settings()
validated['density_percentage'] = max(1, min(100, int(validated.get('density_percentage', 100))))
```

#### UI Controls (`/data/viewer/layout/controls/controls_3d.py`)
Add density slider that's conditionally visible:
```python
# Density Control (only visible when lod_type == 'none')
html.Div([
    html.Label("Density", style={'margin-top': '20px'}),
    dcc.Slider(
        id='density-slider',
        min=1,
        max=100, 
        value=settings['density_percentage'],
        marks={i: f"{i}%" for i in [1, 25, 50, 75, 100]},
        step=1,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
], id='density-controls', style={'display': 'none'})  # Hidden by default
```

#### Callback Updates (`/data/viewer/callbacks/three_d_settings.py`)
1. **Add density slider to update_3d_settings callback**:
   - Add `Input('density-slider', 'value')` to inputs
   - Add `density_percentage` parameter and include in settings dict

2. **Add callback to control density slider visibility**:
   ```python
   @callback(
       outputs=[Output('density-controls', 'style')],
       inputs=[Input('lod-type-dropdown', 'value')],
       group="3d_settings"
   )
   def update_density_controls_visibility(lod_type: str) -> List[Dict[str, str]]:
       """Show density controls only when LOD type is 'none'."""
       if lod_type == 'none':
           return [{'display': 'block', 'margin-top': '20px'}]
       else:
           return [{'display': 'none'}]
   ```

### 3. Density LOD Implementation

#### New DensityLOD Class (`/data/viewer/utils/density_lod.py`)
```python
class DensityLOD:
    """Density-based point cloud subsampling with caching."""
    
    def subsample(
        self, 
        point_cloud_id: str,
        density_percentage: int,
        point_cloud: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Subsample point cloud to specified density percentage."""
        # Cache management and RandomSelect usage
```

#### Integration in Point Cloud Utils (`/data/viewer/utils/point_cloud.py`)
Update `apply_lod_to_point_cloud()`:
```python
def apply_lod_to_point_cloud(
    # ... existing parameters ...
    density_percentage: Optional[int] = None,  # New parameter
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    
    # ... existing validation ...
    
    # Handle density-based subsampling when LOD is 'none'
    if lod_type == "none" and density_percentage is not None and density_percentage < 100:
        from data.viewer.utils.density_lod import DensityLOD
        density_lod = DensityLOD()
        normalized_id = normalize_point_cloud_id(point_cloud_id)
        downsampled = density_lod.subsample(normalized_id, density_percentage, pc_dict)
        # ... return processed data ...
    
    # ... existing LOD logic ...
```

### 4. Display Integration

#### Update Display Functions
Modify display functions to pass density settings:
- `/data/viewer/layout/display/display_pcr.py`
- `/data/viewer/layout/display/display_3dcd.py`

Add `density_percentage` parameter to function calls and pass to `create_point_cloud_figure()`.

#### Backend Integration
Ensure settings are properly synced via existing backend sync mechanism in `/data/viewer/callbacks/backend_sync.py`.

## Implementation Steps

### Phase 1: Core Infrastructure
1. ✅ Analyze existing LOD and caching systems
2. ✅ Design density caching architecture  
3. ✅ Create `DensityLOD` class with caching
4. ✅ Update settings configuration

### Phase 2: UI Integration  
5. ✅ Add density slider to 3D controls
6. ✅ Add visibility control callback
7. ✅ Update settings callback to include density

### Phase 3: Backend Integration
8. ✅ Integrate density processing in `apply_lod_to_point_cloud()`
9. ✅ Update display functions to pass density settings
10. ⏳ Test end-to-end functionality

## Technical Considerations

### Performance
- **Caching Strategy**: Pre-compute common density levels (25%, 50%, 75%) on first access
- **Memory Management**: Implement cache size limits and LRU eviction
- **UI Responsiveness**: Use debouncing for slider changes to avoid excessive re-computation

### User Experience  
- **Intuitive Controls**: Show density as percentage with clear marks
- **Visual Feedback**: Update point cloud title to show "(Density: X%)" when active
- **Smooth Transitions**: Maintain camera position when density changes

### Code Quality
- **Type Safety**: Full type annotations following Pylon conventions
- **Error Handling**: Assertions for input validation, fail-fast on invalid states
- **Documentation**: Comprehensive docstrings and inline comments
- **Testing**: Unit tests for `DensityLOD` class and integration tests

### Integration Points
- **Settings Store**: Leverage existing `3d-settings-store` mechanism
- **Backend Sync**: Use existing sync infrastructure
- **Point Cloud ID**: Reuse existing ID system for cache keys
- **RandomSelect**: Leverage existing `RandomSelect` utility for sampling

## Expected Outcome
Users will have a density slider (1-100%) that appears when "No LOD" is selected, allowing them to control point display density with efficient caching for smooth performance.

## Implementation Summary

### ✅ Completed Implementation

**Core Infrastructure:**
- ✅ **DensityLOD Class**: Created `/data/viewer/utils/density_lod.py` with:
  - Percentage-based subsampling using `RandomSelect`
  - Global caching system (`_global_density_cache`) for performance
  - Reproducible seeding for consistent results
  - Cache management utilities (clear, info, precompute)

**Settings Integration:**
- ✅ **Settings Configuration**: Updated `/data/viewer/utils/settings_config.py`:
  - Added `density_percentage: 100` to default settings
  - Added validation (1-100 range) in `validate_3d_settings()`

**UI Integration:**
- ✅ **Density Slider**: Added to `/data/viewer/layout/controls/controls_3d.py`:
  - 1-100% range with 100 ticks
  - Tooltip showing current percentage
  - Marks at 1%, 25%, 50%, 75%, 100%
  - Hidden by default, shown only when LOD type is 'none'

**Callbacks:**
- ✅ **Settings Callback**: Updated `/data/viewer/callbacks/three_d_settings.py`:
  - Added `density-slider` input to `update_3d_settings()`
  - Added `update_density_controls_visibility()` callback
  - Updated LOD info display to mention density control

**Backend Integration:**
- ✅ **Point Cloud Processing**: Updated `/data/viewer/utils/point_cloud.py`:
  - Added `density_percentage` parameter to `apply_lod_to_point_cloud()`
  - Added `density_percentage` parameter to `create_point_cloud_figure()`
  - Implemented density-based subsampling logic
  - Added density information to figure titles

**Display Functions:**
- ✅ **Display Integration**: Updated display functions:
  - `/data/viewer/callbacks/display.py`: Pass density settings to display functions
  - `/data/viewer/layout/display/display_3dcd.py`: Accept and use density_percentage
  - `/data/viewer/layout/display/display_pcr.py`: Accept and use density_percentage

### How It Works

1. **UI Control**: When LOD type is set to "none", the density slider appears
2. **Settings Flow**: Slider changes → 3D settings store → backend sync → display functions
3. **Processing**: Display functions call `create_point_cloud_figure()` with `density_percentage`
4. **Caching**: `DensityLOD` caches subsampled point clouds by ID and percentage
5. **Performance**: Camera changes use cached data; only density changes trigger recomputation

### Ready for Testing

The implementation is complete and ready for end-to-end testing. The density control should:
- Appear when "No LOD" is selected
- Allow 1-100% density control with smooth slider interaction
- Cache subsampled point clouds for performance
- Update point cloud visualizations in real-time
- Show density information in figure titles