from typing import Dict


def base_container_style() -> Dict[str, str]:
    return {
        'width': '100%',
        'height': '100%',
        'display': 'flex',
        'flexDirection': 'column',
        'position': 'relative',
    }
