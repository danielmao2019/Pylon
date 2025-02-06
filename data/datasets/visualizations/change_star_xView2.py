import sys
sys.path.append("../../")
import os
import random
import torch
import matplotlib.pyplot as plt
import data


if __name__ == "__main__":
    dataset = data.datasets.xView2Dataset(
        data_root="./soft_links/xView2",
        split="train",
    )
    collate_fn = data.collators.ChangeStarCollator(method="train")
    batch_size = 10
    batch = collate_fn([dataset[idx] for idx in random.sample(range(len(dataset)), batch_size)])
    output_dir = f"./{dataset.__class__.__name__}"
    if os.path.isdir(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    inputs, labels, meta_info = batch['inputs'], batch['labels'], batch['meta_info']
    for idx in range(batch_size):
        img_1 = inputs['img_1'][idx]
        img_2 = inputs['img_2'][idx]
        change_map = labels['change'][idx]
        img_1 = (img_1.permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy()  # (H, W, C)
        img_2 = (img_2.permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy()  # (H, W, C)
        change_map = (change_map * 255).cpu().numpy()  # (H, W)
        fig, axes = plt.subplots(1, 3, figsize=(3*4, 1*4))
        axes[0].imshow(img_1)
        axes[0].set_title("Image 1")
        axes[0].axis("off")

        axes[1].imshow(img_2)
        axes[1].set_title("Image 2")
        axes[1].axis("off")

        axes[2].imshow(change_map, cmap="gray")
        axes[2].set_title("Change Map")
        axes[2].axis("off")

        # Save the figure
        save_path = os.path.join(output_dir, f"datapoint_{idx}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
