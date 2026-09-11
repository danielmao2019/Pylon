import argparse

import torch
from scene_rendering import (
    DEVICES,
    POINT_SIZES,
    RENDERERS,
    RETURN_MASK_OPTIONS,
    build_camera,
    build_point_cloud,
    render_single_camera,
)

from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import (
    create_circular_kernel_offsets,
)


def main() -> None:
    """Renders every scene with main's code in a child process launched inside the main checkout, so the branch has a fixed reference to compare against.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Render the saved scenes with the checkout this process imports from."
    )
    parser.add_argument(
        "--scenes_path", type=str, required=True, help="Path of the saved scenes."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path the renders are saved to."
    )
    args = parser.parse_args()
    # main resolves a shared pixel by which write lands last, and deterministic mode makes that the last write in point order on cpu and cuda alike.
    torch.use_deterministic_algorithms(True)
    scenes = torch.load(args.scenes_path)

    # --- One render per device, scene, camera, renderer, point size and mask option
    renders = {}
    for device, scene, camera_index, renderer, point_size, return_mask in (
        (device, scene, camera_index, renderer, point_size, return_mask)
        for device in DEVICES
        for scene in scenes
        for camera_index in range(len(scene["cameras"]))
        for renderer in RENDERERS
        for point_size in POINT_SIZES
        for return_mask in RETURN_MASK_OPTIONS
    ):
        pc = build_point_cloud(scene=scene, device=device)
        camera = build_camera(scene=scene, camera_index=camera_index, device=device)
        output = render_single_camera(
            renderer=renderer,
            pc=pc,
            camera=camera,
            resolution=scene["resolution"],
            return_mask=return_mask,
            point_size=point_size,
        )
        renders[
            (
                str(device),
                scene["name"],
                camera_index,
                renderer,
                point_size,
                return_mask,
            )
        ] = (
            tuple(member.cpu() for member in output) if return_mask else output.cpu()
        )

    # --- main's kernel offsets per point size
    kernels = {}
    for point_size in POINT_SIZES:
        kernels[point_size] = create_circular_kernel_offsets(
            point_size=point_size, device=torch.device("cpu")
        )

    torch.save({"renders": renders, "kernels": kernels}, args.output_path)


if __name__ == "__main__":
    main()
