from typing import Optional
import torch
import data


class BaseDataLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        last_mode: Optional[str] = 'keep',
        **kwargs,
    ) -> None:
        """
        Custom DataLoader with `last_mode`:
        - 'drop': drop the last incomplete batch (equivalent to drop_last=True in PyTorch DataLoader)
        - 'keep': keep the last incomplete batch (equivalent to drop_last=False)
        - 'fill': fill the last incomplete batch by sampling with replacement
        """
        assert last_mode in {'drop', 'keep', 'fill'}, \
            f"last_mode must be one of {'drop', 'keep', 'fill'}. Got {last_mode}."
        
        if last_mode == 'fill':
            sampler = data.samplers.FillLastBatchSampler(dataset, batch_size, shuffle=shuffle)
            for key in ['batch_size', 'shuffle', 'sampler', 'drop_last']:
                if key in kwargs:
                    del kwargs[key]
            super(BaseDataLoader, self).__init__(dataset, batch_sampler=sampler, **kwargs)
        else:
            drop_last = (last_mode == 'drop')
            super(BaseDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
