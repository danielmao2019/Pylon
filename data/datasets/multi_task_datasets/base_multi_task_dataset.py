from abc import ABC
from data.datasets.base_dataset import BaseDataset


class BaseMultiTaskDataset(BaseDataset, ABC):
    """Base class for multi-task learning datasets.
    
    This class serves as a marker base class for all multi-task datasets,
    enabling clean type detection in the viewer backend. It inherits from
    BaseDataset and doesn't need to implement additional methods since
    multi-task datasets are diverse and don't share common functionality
    beyond the standard BaseDataset interface.
    
    All multi-task datasets should inherit from this class to ensure
    proper integration with the data viewer system.
    """
    pass