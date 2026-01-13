from abc import ABC
from typing import Optional, List, Dict, Any
from data.datasets.base_dataset import BaseDataset


class BaseMultiTaskDataset(BaseDataset, ABC):
    """Base class for multi-task learning datasets.

    This class serves as a marker base class for all multi-task datasets,
    enabling clean type detection in the viewer backend. It also provides
    selective label loading functionality to optimize performance when only
    a subset of tasks is needed.

    All multi-task datasets should inherit from this class to ensure
    proper integration with the data viewer system.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        *args,
        **kwargs
    ) -> None:
        """Initialize multi-task dataset with optional selective label loading.

        Args:
            labels: Optional list of label names to load. Must be a subset of
                   LABEL_NAMES. If None, all labels will be loaded.
            *args, **kwargs: Arguments passed to BaseDataset
        """
        # Basic validation before super().__init__()
        if labels is not None:
            assert isinstance(labels, list), f"labels must be list, got {type(labels)}"
            assert all(isinstance(label, str) for label in labels), "All labels must be strings"
            assert len(labels) > 0, "labels list must not be empty"

        # Set selected_labels BEFORE super().__init__() so it's available during cache initialization
        # We need to validate against LABEL_NAMES, but LABEL_NAMES should be available as a class attribute
        if labels is not None:
            label_set = set(labels)
            available_set = set(self.LABEL_NAMES)
            assert label_set.issubset(available_set), (
                f"labels {label_set - available_set} not in LABEL_NAMES {available_set}"
            )

            self.selected_labels = labels
        else:
            self.selected_labels = self.LABEL_NAMES.copy()

        super().__init__(*args, **kwargs)

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()

        # Include selected labels in cache versioning since we now load different
        # data from disk based on selected labels
        version_dict['selected_labels'] = sorted(self.selected_labels)

        return version_dict
