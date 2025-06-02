from typing import Tuple
import torch
import os
import json


torch.manual_seed(0)
gt = torch.rand(size=(2, 2), dtype=torch.float32)


class Dataset(torch.utils.data.Dataset):

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(idx)
        x = torch.rand(size=(2,), dtype=torch.float32)
        noise = torch.randn(size=(2,), dtype=torch.float32)
        y = gt @ x + noise * 0.001
        return {'x': x, 'y': y}

    def __len__(self) -> int:
        return 10


def run_pytorch() -> None:
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = torch.nn.Linear(in_features=2, out_features=2)
    criterion = torch.nn.MSELoss(reduction='mean')
    metric = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-03)
    work_dir = "./logs/tests/supervised_single_task_trainer/compare_pytorch"
    for idx in range(100):
        all_losses = []
        all_scores = []
        # train epoch
        model.train()
        for datapoint in dataloader:
            loss = criterion(input=model(datapoint[0]), target=datapoint[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
        # save losses
        save_losses = torch.tensor(all_losses)
        torch.save(save_losses, os.path.join(work_dir, f"epoch_{idx}", "training_losses.pt"))
        # val epoch
        model.eval()
        with torch.no_grad():
            for datapoint in dataloader:
                score = metric(input=model(datapoint[0]), target=datapoint[1])
                all_scores.append(score.item())
        # save scores
        save_scores = {
            'aggregated': {
                'score': torch.tensor(all_scores).detach().mean().item(),
            },
            'per_datapoint': {
                'score': torch.tensor(all_scores).detach().tolist(),
            },
        }
        with open(os.path.join(work_dir, f"epoch_{idx}", "validation_scores.pt"), "w") as f:
            json.dump(save_scores, f)


def test_compare_pytorch() -> None:
    run_pytorch()
    dir1 = "./logs/examples/linear/"
    dir2 = "./logs/tests/supervised_single_task_trainer/compare_pytorch"
    for idx in range(100):
        losses1 = torch.load(os.path.join(dir1, f"epoch_{idx}", "training_losses.pt"))
        losses2 = torch.load(os.path.join(dir2, f"epoch_{idx}", "training_losses.pt"))
        assert torch.allclose(losses1, losses2), f"{idx} - losses"
        scores1 = json.load(open(os.path.join(dir1, f"epoch_{idx}", "validation_scores.pt")))
        scores2 = json.load(open(os.path.join(dir2, f"epoch_{idx}", "validation_scores.pt")))
        assert scores1 == scores2, f"{idx} - scores"
