# Data Viewer Gaussians Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/gaussians/
├── dash/
│   ├── core_gaussians_display.py  # Dash Gaussian display object core
│   └── apis.py                    # Dash Gaussian-display APIs
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend Gaussian-display response schemas: url + meta_info
    │   ├── core_gaussians_display.py  # TS DisplayResponse core for Gaussians
    │   └── apis.py                    # TS backend Gaussian-display APIs
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS Gaussian-display response interfaces: url + meta_info
        ├── core_gaussians_display.ts  # TS Gaussian UI core
        └── apis.ts                    # TS frontend Gaussian-display APIs
```
