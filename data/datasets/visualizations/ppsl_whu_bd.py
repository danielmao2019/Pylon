import sys
sys.path.append("../../")
import os
import data


if __name__ == "__main__":
    source_dataset = data.datasets.WHU_BD_Dataset(data_root="./soft_links/WHU-BD", split='train')
    dataset = data.datasets.PPSLDataset(
        source=source_dataset,
        dataset_size=len(source_dataset),
    )
    output_dir = f"./{dataset.__class__.__name__}"
    if os.path.isdir(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    dataset.visualize(output_dir)
