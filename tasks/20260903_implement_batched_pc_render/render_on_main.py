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

from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import (
    apply_point_size_postprocessing,
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
    # main resolves a shared pixel by which write lands last, and deterministic mode makes that outcome reproducible run to run.
    torch.use_deterministic_algorithms(True)
    scenes = torch.load(args.scenes_path)

    # --- One render per device, scene, camera, renderer, point size and mask option
    renders = {}
    for device in DEVICES:
        for scene in scenes:
            for camera_index in range(len(scene["cameras"])):
                for renderer in RENDERERS:
                    for point_size in POINT_SIZES:
                        for return_mask in RETURN_MASK_OPTIONS:
                            pc = build_point_cloud(scene=scene, device=device)
                            camera = build_camera(
                                scene=scene, camera_index=camera_index, device=device
                            )
                            output = render_single_camera(
                                renderer=renderer,
                                pc=pc,
                                camera=camera,
                                resolution=scene["resolution"],
                                return_mask=return_mask,
                                point_size=point_size,
                            )
                            if return_mask:
                                # The map and its mask.
                                renders[
                                    (
                                        str(device),
                                        scene["name"],
                                        camera_index,
                                        renderer,
                                        point_size,
                                        return_mask,
                                    )
                                ] = (output[0].cpu(), output[1].cpu())
                            else:
                                renders[
                                    (
                                        str(device),
                                        scene["name"],
                                        camera_index,
                                        renderer,
                                        point_size,
                                        return_mask,
                                    )
                                ] = output.cpu()

    # --- main's kernel offsets per point size
    kernels = {}
    for point_size in POINT_SIZES:
        kernels[point_size] = create_circular_kernel_offsets(
            point_size=point_size, device=torch.device("cpu")
        )

    # --- main's dilation of each camera's depth render at every point size above one
    dilations = {}
    for device in DEVICES:
        for scene in scenes:
            for camera_index in range(len(scene["cameras"])):
                for point_size in POINT_SIZES:
                    if point_size > 1.0:
                        # main's depth entry fills the pixels no point lands on with its default ignore_value of -1.0.
                        depth_map = renders[
                            (
                                str(device),
                                scene["name"],
                                camera_index,
                                "depth",
                                1.0,
                                False,
                            )
                        ].masked_fill(
                            renders[
                                (
                                    str(device),
                                    scene["name"],
                                    camera_index,
                                    "depth",
                                    1.0,
                                    False,
                                )
                            ]
                            == -1.0,
                            float("inf"),
                        )
                        dilations[
                            (str(device), scene["name"], camera_index, point_size)
                        ] = apply_point_size_postprocessing(
                            rendered_image=depth_map,
                            depth_map=depth_map,
                            point_size=point_size,
                            ignore_value=float("inf"),
                        )

    torch.save(
        {"renders": renders, "kernels": kernels, "dilations": dilations},
        args.output_path,
    )


if __name__ == "__main__":
    main()
