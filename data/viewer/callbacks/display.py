"""Display utility module for the viewer.

This module previously contained display creation utilities, but has been refactored
to follow the project's "fail fast and loud" philosophy. The actual display update 
callbacks are distributed across:
- dataset.py: Dataset selection triggered display updates
- transforms.py: Transform checkbox triggered display updates  
- navigation.py: Datapoint navigation triggered display updates

Display functions are now called directly without defensive wrapper functions.
"""
