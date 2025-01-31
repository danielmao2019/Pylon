import sys
sys.path.append("../../")
import os
import data


if __name__ == "__main__":
    source_dataset = data.datasets.Bi2SingleTemporal(
        source=data.datasets.SYSU_CD_Dataset(data_root="./soft_links/SYSU-CD", split="train"),
    )
    dataset = data.datasets.I3PEDataset(
        source=source_dataset,
        dataset_size=len(source_dataset),
        exchange_ratio=0.75,
    )
    output_dir = f"./visualization/{dataset.__class__.__name__}"
    if os.path.isdir(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    dataset.visualize(output_dir)
