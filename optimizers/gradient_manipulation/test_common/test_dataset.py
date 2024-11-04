import torch


class GradientManipulationTestDataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        super(GradientManipulationTestDataset, self).__init__()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> torch.Tensor:
        torch.manual_seed(idx)
        inputs = torch.randn(size=(2,))
        labels = {
            'task1': torch.randn(size=(2,)),
            'task2': torch.randn(size=(2,)),
        }
        dp = {
            'inputs': inputs,
            'labels': labels,
        }
        return dp
