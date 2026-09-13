# Data Viewer AABBs Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/aabbs/
├── threed/
│   └── ts/
│       ├── backend/
│       │   ├── schemas/
│       │   │   └── display_response.py  # Aabb3dDisplayResponse: inline 3D boxes + optional per-box scores
│       │   └── apis.py  # create_aabb_3d_display_response
│       └── frontend/
│           ├── types/
│           │   └── display_response.ts  # Aabb3dDisplayResponse interface
│           └── apis.ts  # renderAabb3dDisplay (standalone) + createAabb3dObject (part-B) for the spatial 3D boxes + score labels; self-registers aabb_3d
└── twod/
    └── ts/
        ├── backend/
        │   ├── schemas/
        │   │   └── display_response.py  # Aabb2dDisplayResponse: inline 2D boxes + optional per-box scores
        │   └── apis.py  # create_aabb_2d_display_response
        └── frontend/
            ├── types/
            │   └── display_response.ts  # Aabb2dDisplayResponse interface
            └── apis.ts  # renderAabb2dDisplay: raster overlay of 2D boxes + score labels
```
