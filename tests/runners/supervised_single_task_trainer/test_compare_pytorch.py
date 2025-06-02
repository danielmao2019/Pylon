from typing import Tuple
import torch
import os
import json


torch.manual_seed(0)
gt = torch.rand(size=(2, 2), dtype=torch.float32)


class Dataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        self.generator = torch.Generator()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.generator.manual_seed(idx)
        x = torch.rand(size=(2,), dtype=torch.float32, generator=self.generator)
        noise = torch.randn(size=(2,), dtype=torch.float32, generator=self.generator)
        y = gt @ x + noise * 0.001
        return x, y

    def __len__(self) -> int:
        return 10


def run_pytorch() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)
    train_dataset = Dataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    val_dataset = Dataset()
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8)
    criterion = torch.nn.MSELoss(reduction='mean')
    metric = torch.nn.MSELoss(reduction='mean')
    model = torch.nn.Linear(in_features=2, out_features=2)
    optimizer = torch.optim.SGD(params=list(model.parameters()), lr=1e-03)
    work_dir = "./logs/tests/supervised_single_task_trainer/compare_pytorch"
    os.system(f"rm -rf {work_dir}")
    os.makedirs(work_dir, exist_ok=True)
    for idx in range(10):
        torch.manual_seed(0)
        all_losses = []
        all_scores = []
        epoch_dir = os.path.join(work_dir, f"epoch_{idx}")
        os.makedirs(epoch_dir, exist_ok=True)
        # train epoch
        model.train()
        for datapoint in train_dataloader:
            output = model(datapoint[0])
            loss = criterion(input=output, target=datapoint[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
        # save losses
        save_losses = torch.tensor(all_losses)
        torch.save(save_losses, os.path.join(epoch_dir, "training_losses.pt"))
        # val epoch
        model.eval()
        with torch.no_grad():
            for datapoint in val_dataloader:
                output = model(datapoint[0])
                score = metric(input=output, target=datapoint[1])
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
        with open(os.path.join(epoch_dir, "validation_scores.json"), "w") as f:
            json.dump(save_scores, f)


def test_compare_pytorch() -> None:
    run_pytorch()
    dir1 = "./logs/examples/linear/"
    dir2 = "./logs/tests/supervised_single_task_trainer/compare_pytorch"
    for idx in range(10):
        losses1 = torch.load(os.path.join(dir1, f"epoch_{idx}", "training_losses.pt"))
        losses2 = torch.load(os.path.join(dir2, f"epoch_{idx}", "training_losses.pt"))
        assert torch.allclose(losses1, losses2), f"{idx} - losses"
        scores1 = json.load(open(os.path.join(dir1, f"epoch_{idx}", "validation_scores.json")))
        scores2 = json.load(open(os.path.join(dir2, f"epoch_{idx}", "validation_scores.json")))
        assert scores1 == scores2, f"{idx} - scores"
