# Debouncing Implementation Plan for Pylon Data Viewer

## Problem Statement
The data viewer currently experiences performance issues during rapid user interactions, particularly when:
- Dragging sliders (navigation, point size, opacity, radius, density)
- Toggling multiple transform checkboxes quickly
- Adjusting 3D visualization settings

Each interaction triggers immediate callbacks that fetch data from the backend and re-render complex visualizations, causing lag and poor user experience.

## Solution Overview
Implement a debouncing mechanism that delays callback execution until user interactions have stabilized, preventing unnecessary intermediate updates while maintaining responsive final updates.

## Implementation Strategy

### 1. Create Debouncing Infrastructure

#### 1.1 Debounce Decorator
Create a new module `data/viewer/utils/debounce.py` with:
- A `debounce` decorator that wraps callback functions
- Configurable delay parameter (default 250-500ms)
- Support for both single and grouped debouncing
- Thread-safe implementation using locks

#### 1.2 Client-Side Debouncing
Add client-side JavaScript debouncing for immediate UI feedback:
- Create `data/viewer/assets/debounce.js`
- Implement debouncing for slider drag events
- Show loading indicators during debounce period

### 2. Apply Debouncing to Critical Callbacks

#### 2.1 Navigation Callbacks (Highest Priority)
- `update_datapoint_from_navigation()` - Debounce by 300ms
- `update_index_from_buttons()` - No debouncing (discrete clicks)
- `update_current_index()` - Immediate (lightweight UI update)

#### 2.2 3D Settings Callbacks (High Priority)
- `update_3d_settings()` - Debounce by 250ms
- Group all 3D setting inputs to debounce together
- Prevent cascading updates during adjustment

#### 2.3 Transform Callbacks (Medium Priority)
- `update_datapoint_from_transforms()` - Debounce by 400ms
- Allow rapid checkbox toggling without immediate re-renders

#### 2.4 Camera Sync Callbacks (Low Priority)
- Keep existing implementation (already efficient)

### 3. Implementation Details

#### 3.1 Debounce Decorator Design
```python
@debounce(delay_ms=300, group_id="navigation")
def update_datapoint_from_navigation(...):
    # Existing implementation
```

Features:
- Per-callback or grouped debouncing
- Cancellable pending executions
- Last-call-wins semantics
- Proper error handling and logging

#### 3.2 UI Feedback Mechanisms
- Loading spinners during debounce delay
- Temporary opacity reduction on elements being updated
- Smooth transitions between states

#### 3.3 Configuration Management
Add debounce settings to `ViewerSettings`:
- Configurable delays per callback type
- Enable/disable debouncing globally
- Performance profiling options

### 4. Testing Strategy

#### 4.1 Unit Tests
- Test debounce decorator with various delays
- Test cancellation and grouping behavior
- Test thread safety

#### 4.2 Integration Tests
- Test rapid slider movements
- Test mixed interactions (sliders + checkboxes)
- Test that final states are correct

#### 4.3 Performance Tests
- Measure callback frequency before/after
- Measure UI responsiveness improvements
- Profile memory usage during rapid interactions

### 5. Rollback Strategy
- Debouncing can be disabled via configuration
- Original callbacks remain intact (decorator can be removed)
- No changes to core data flow or business logic

## Benefits
1. **Improved Performance**: Reduce callback executions by 80-90% during rapid interactions
2. **Better UX**: Smoother, more responsive interface
3. **Resource Efficiency**: Less CPU/memory usage, reduced backend load
4. **Maintainability**: Clean separation of debouncing logic from business logic

## Risks and Mitigations
1. **Risk**: Users might perceive delay as lag
   - **Mitigation**: Add immediate UI feedback (loading indicators)
   
2. **Risk**: Final update might be missed
   - **Mitigation**: Ensure last call always executes
   
3. **Risk**: Complex interactions might not debounce correctly
   - **Mitigation**: Comprehensive testing and monitoring

## Next Steps
1. Review and approve this plan
2. Implement debounce decorator
3. Apply to navigation callbacks first (highest impact)
4. Test and measure improvements
5. Roll out to other callback groups
6. Add configuration options