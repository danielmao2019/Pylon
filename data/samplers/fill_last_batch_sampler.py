from torch.utils.data import Dataset, Sampler
import random


class FillLastBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # If there's an incomplete batch, fill it by sampling with replacement
        if batch:
            while len(batch) < self.batch_size:
                batch.append(random.choice(self.indices))  # Sample with replacement
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # Round up
