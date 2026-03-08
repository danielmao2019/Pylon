from abc import ABC
from typing import Any, Dict, List, Optional

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
        **kwargs: Any,
    ) -> None:
        """Initialize multi-task dataset with optional selective label loading.

        Args:
            labels: Optional list of label names to load. Must be a subset of
                   LABEL_NAMES. If None, all labels will be loaded.
            *args, **kwargs: Arguments passed to BaseDataset
        """
        # Input validations
        assert labels is None or isinstance(labels, list), f"{type(labels)=}"
        assert labels is None or all(
            isinstance(label, str) for label in labels
        ), f"{labels=}"
        assert isinstance(kwargs, dict), f"{type(kwargs)=}"
        assert hasattr(self, "LABEL_NAMES"), f"{self.__class__.__name__=}"
        assert isinstance(self.LABEL_NAMES, list), f"{type(self.LABEL_NAMES)=}"
        assert len(self.LABEL_NAMES) > 0, f"{self.LABEL_NAMES=}"
        assert all(
            isinstance(label_name, str) for label_name in self.LABEL_NAMES
        ), f"{self.LABEL_NAMES=}"
        assert labels is None or set(labels).issubset(
            set(self.LABEL_NAMES)
        ), f"{labels=} {self.LABEL_NAMES=}"

        # Class attributes
        self.selected_labels = self.LABEL_NAMES if labels is None else labels

        super().__init__(**kwargs)

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()

        # Include selected labels in cache versioning since we now load different
        # data from disk based on selected labels
        version_dict["selected_labels"] = sorted(self.selected_labels)

        return version_dict
