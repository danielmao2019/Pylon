import sys
sys.path.append("../../")
import os
import data


if __name__ == "__main__":
    dataset = data.datasets.CDDDataset(data_root="./soft_links/CDD", split='train')
    output_dir = f"./visualizations/{dataset.__class__.__name__}"
    if os.path.isdir(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    dataset.visualize(output_dir)
