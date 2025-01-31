import sys
sys.path.append("../../")
import data


if __name__ == "__main__":
    source_dataset = data.datasets.Bi2SingleTemporal(
        source=data.datasets.SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split="train"),
    )
    dataset = data.datasets.I3PEDataset(
        source=source_dataset,
        dataset_size=len(source_dataset),
        exchange_ratio=0.75,
    )
    output_dir = f"./{dataset.__class__}_visualization"
    dataset.visualize(output_dir)
