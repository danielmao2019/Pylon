# Simplified Debouncing Implementation Plan for Pylon Data Viewer

## Problem Statement
The data viewer currently experiences performance issues during rapid user interactions, particularly when:
- **Dragging/rotating 3D visualizations** (camera manipulation)
- **Dragging sliders** (navigation, point size, opacity, radius, density)
- **Rapidly clicking navigation buttons** (prev/next)
- **Toggling multiple transform checkboxes quickly**

Each interaction triggers immediate callbacks that fetch data from the backend and re-render complex visualizations, causing lag and poor user experience.

## Solution Overview
Implement a simple debouncing mechanism with a fixed 0.5 second delay for all callbacks, preventing excessive recomputation during rapid interactions.

## Implementation Strategy

### 1. Create Debouncing Infrastructure

#### 1.1 Debounce Decorator
Create a new module `data/viewer/utils/debounce.py`:
- Simple `@debounce` decorator with fixed 0.5s delay
- Thread-safe implementation using threading.Timer
- Last-call-wins semantics (only final interaction executes)
- No error handling (fail fast philosophy)

### 2. Apply Debouncing to Callbacks (Priority Order)

#### 2.1 3D Camera Manipulation (Highest Priority)
- Camera rotation/zoom/pan callbacks
- Most frequent source of rapid events

#### 2.2 Navigation Controls (Highest Priority)
- `update_datapoint_from_navigation()` - Slider dragging
- `update_index_from_buttons()` - Rapid button clicks
- `update_current_index()` - Index display updates

#### 2.3 3D Settings Sliders (High Priority)
- `update_3d_settings()` - Point size, opacity, radius, density
- All 3D visualization parameters

#### 2.4 Transform Checkboxes (Medium Priority)
- `update_datapoint_from_transforms()` - Transform toggles
- Allow rapid checkbox clicking without re-renders

### 3. Implementation Details

#### 3.1 Debounce Decorator Design
```python
@debounce
def update_datapoint_from_navigation(...):
    # Existing implementation
```

Features:
- Fixed 0.5 second delay for all callbacks
- Single debouncing only (no grouping)
- Cancel previous timer on new call
- Execute only the final call

#### 3.2 No Client-Side JavaScript
- Pure Python server-side solution
- Avoid JS/WebGL complexity
- Rely on Dash's existing client-server communication

### 4. Testing Strategy

#### 4.1 Unit Tests (tests/data/viewer/)
Create comprehensive unit tests in `tests/data/viewer/test_debounce.py`:
- Test single rapid call execution
- Test multiple rapid calls (only last executes)
- Test timing (0.5s delay is respected)
- Test argument preservation
- Test concurrent debouncing (multiple functions)
- Test thread safety

All unit tests will be implemented and verified to pass.

#### 4.2 Manual Testing (Left to User)
- Test slider dragging behavior
- Test 3D visualization manipulation
- Test rapid button clicking
- Verify overall UI responsiveness

### 5. Rollback Strategy
- Debouncing is applied via decorator (easy to remove)
- No changes to core callback logic
- Original behavior restored by removing @debounce

## Benefits
1. **Improved Performance**: Reduce callback executions by 80-90% during rapid interactions
2. **Simple Implementation**: Fixed delay, no configuration complexity
3. **Resource Efficiency**: Less CPU/memory usage, reduced backend load
4. **Easy Testing**: Clear unit tests, simple manual verification

## Implementation Order
1. Implement debounce decorator in `data/viewer/utils/debounce.py`
2. Write and verify unit tests in `tests/data/viewer/test_debounce.py`
3. Apply decorator to callbacks in priority order
4. Manual testing and verification

## Next Steps
1. Review and approve this simplified plan
2. Implement debounce decorator with unit tests
3. Apply to highest priority callbacks first
4. Perform manual testing
5. Iterate based on results
