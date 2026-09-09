from typing import List, Union

import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud


class Select:
    """Indexes a point cloud down to the points a fixed index list or index tensor names."""

    def __init__(self, indices: Union[torch.Tensor, List[int]]) -> None:
        """Holds the indices this selection will take, in the list or tensor form it was given.

        Args:
            indices: The point indices to take, as a torch tensor of shape [K] carrying torch.int64 or as a list of K ints.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            assert isinstance(
                indices, (torch.Tensor, list)
            ), f"indices arrive as a torch tensor or a list: type(indices)={type(indices)}"
            if isinstance(indices, list):
                # a tensor's dtype and device are checked where the point cloud's device is known, which is not here
                assert all(
                    isinstance(index, int) for index in indices
                ), f"every entry of an index list is an int: index types={tuple(type(index) for index in indices)}"

        _validate_inputs()

        self.indices = indices

    def __call__(self, pc: PointCloud) -> PointCloud:
        """Builds a new point cloud carrying every field of pc indexed down to the selected points, the meta data travelling across whole and each field's current dtype with it.

        Args:
            pc: The point cloud to index, whose fields are torch tensors of shape [N] or [N, C] on one device.

        Returns:
            The selected point cloud, carrying every field of pc indexed down to the K selected points plus an 'indices' field naming which points of pc they were.
        """

        def _validate_inputs() -> None:
            # a reusable door, so it asserts what it needs whoever calls it
            assert isinstance(
                pc, PointCloud
            ), f"a selection indexes a point cloud: type(pc)={type(pc)}"

        _validate_inputs()

        indices = self._materialize_indices(device=pc.device)
        assert bool(
            (indices < pc.num_points).all()
        ), f"every selected index names a point the cloud carries: num_points={pc.num_points}, out-of-range indices={indices[indices >= pc.num_points].tolist()}"
        fields = {}
        for name in pc.field_names():
            # an indices field the cloud already carries is data like any other here, since the indices this selection takes are its own
            fields[name] = getattr(pc, name)[indices]
        if 'indices' not in fields:
            # this selection made the field, so no source column stands behind it and the meta data crossing over names it nowhere
            fields['indices'] = indices
        # a selection is not a source, so the record crosses whole rather than being derived again from the indexed tensors
        return PointCloud(data=fields, meta_data=pc.meta_data)

    def __str__(self) -> str:
        """Renders the selection, spelling the indices out only while there are at most five of them.

        Args:
            None.

        Returns:
            The rendering, as a str naming either the indices themselves or how many of them there are.
        """
        num_indices = (
            len(self.indices)
            if isinstance(self.indices, list)
            else self.indices.numel()
        )
        if num_indices <= 5:
            return f"Select(indices={self.indices if isinstance(self.indices, list) else self.indices.tolist()})"
        return f"Select(indices=[...{num_indices} indices])"

    def _materialize_indices(self, device: torch.device) -> torch.Tensor:
        """Turns the held indices into a non-negative int64 tensor sitting on the point cloud's device.

        Args:
            device: The torch device the point cloud's fields sit on, which the index tensor must sit on too.

        Returns:
            The indices, as a torch tensor of shape [K] carrying torch.int64 on device, every entry non-negative.
        """
        if isinstance(self.indices, list):
            indices_tensor = torch.tensor(
                self.indices, dtype=torch.int64, device=device
            )
        else:
            assert (
                self.indices.dtype == torch.int64
            ), f"an index tensor carries torch.int64: indices.dtype={self.indices.dtype}"
            assert (
                self.indices.device == device
            ), f"an index tensor sits on the point cloud's device: indices.device={self.indices.device}, device={device}"
            indices_tensor = self.indices
        assert bool(
            (indices_tensor >= 0).all()
        ), f"every index is non-negative: negative indices={indices_tensor[indices_tensor < 0].tolist()}"
        return indices_tensor
