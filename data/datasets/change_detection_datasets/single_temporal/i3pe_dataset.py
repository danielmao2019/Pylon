from typing import Tuple, Dict, Any
import random
import numpy
import torch
from sklearn.cluster import DBSCAN
from skimage.segmentation import slic
from data.datasets import BaseSyntheticDataset


class I3PEDataset(BaseSyntheticDataset):
    __doc__ = r"""
    References:
        * https://github.com/ChenHongruixuan/I3PE/blob/3182c2918bd32a5b7dd44dc7ee71fe09ab92ed7b/data/datasets.py#L67
        * https://github.com/ChenHongruixuan/I3PE/blob/3182c2918bd32a5b7dd44dc7ee71fe09ab92ed7b/data/generate_clustering_results.py#L51
        * https://github.com/ChenHongruixuan/I3PE/blob/3182c2918bd32a5b7dd44dc7ee71fe09ab92ed7b/data/generate_object.py#L10
    """

    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    n_segments = 1000
    eps = 7.0
    min_samples = 10
    scale_factors = [16, 32, 64, 128]

    def __init__(self, exchange_ratio: float, **kwargs: Any) -> None:
        """
        Initialize the I3PEDataset.

        Args:
            exchange_ratio (float): Ratio of patches to exchange.
            **kwargs (Any): Additional arguments passed to the BaseSyntheticDataset.
        """
        self.exchange_ratio = exchange_ratio
        super(I3PEDataset, self).__init__(**kwargs)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        exchange_type_seed = random.random()
        patch_size = random.choice(self.scale_factors)

        if exchange_type_seed < 0.5:
            idx_2 = idx
            img_1 = self.dataset[idx]['inputs']['image']
            class_labels = self._perform_clustering(img_1)
            img_2, change_map = self._intra_image_patch_exchange(img_1, class_labels, patch_size)
        else:
            idx_2 = random.choice(range(len(self.dataset)))
            img_1 = self.dataset[idx]['inputs']['image']
            img_2 = self.dataset[idx_2]['inputs']['image']
            objects_1 = self._segment_objects(img_1)
            objects_2 = self._segment_objects(img_2)
            img_2, change_map = self._inter_image_patch_exchange(img_1, img_2, objects_1, objects_2, patch_size)

        inputs = {
            'img_1': img_1,
            'img_2': img_2,
        }
        labels = {
            'change_map': change_map,
        }
        meta_info = {
            'img_1_filepath': self.dataset[idx]['meta_info']['image_filepath'],
            'img_2_filepath': self.dataset[idx_2]['meta_info']['image_filepath'],
            'patch_size': patch_size,
        }
        return inputs, labels, meta_info

    def _segment_objects(self, image: torch.Tensor) -> numpy.ndarray:
        """
        Perform object segmentation using SLIC.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            numpy.ndarray: Object segmentation map.
        """
        return slic(image.numpy(), n_segments=self.n_segments, start_label=0)

    def _perform_clustering(self, image: torch.Tensor) -> numpy.ndarray:
        """
        Perform clustering on image segments using DBSCAN.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            numpy.ndarray: Clustered labels map.
        """
        segments = self._segment_objects(image)
        num_segments = numpy.max(segments)

        features = numpy.zeros((num_segments + 1, 6))
        for segment_idx in range(num_segments + 1):
            segment_mask = segments == segment_idx
            features[segment_idx] = numpy.concatenate([
                numpy.mean(image[segment_mask], axis=0),
                numpy.std(image[segment_mask], axis=0)
            ])

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean').fit(features)
        cluster_labels = clustering.labels_

        clustered_map = numpy.zeros(image.shape[:2], dtype=int)
        for segment_idx in range(num_segments + 1):
            clustered_map[segments == segment_idx] = cluster_labels[segment_idx] + 1

        return clustered_map

    def _intra_image_patch_exchange(self, image: torch.Tensor, labels: numpy.ndarray, patch_size: int) -> Tuple[torch.Tensor, numpy.ndarray]:
        """
        Perform intra-image patch exchange.

        Args:
            image (torch.Tensor): Input image tensor.
            labels (numpy.ndarray): Clustering labels map.
            patch_size (int): Size of patches to exchange.

        Returns:
            Tuple[torch.Tensor, numpy.ndarray]: Modified image and change label map.
        """
        num_patches = image.shape[0] // patch_size
        patch_indices = numpy.arange(num_patches ** 2)
        numpy.random.shuffle(patch_indices)

        num_exchanges = int(num_patches ** 2 * self.exchange_ratio)
        exchanged_image = image.clone()
        change_map = numpy.zeros(image.shape[:2], dtype=numpy.uint8)

        for i in range(0, num_exchanges, 2):
            idx_1 = numpy.unravel_index(patch_indices[i], (num_patches, num_patches))
            idx_2 = numpy.unravel_index(patch_indices[i + 1], (num_patches, num_patches))

            patch_1 = image[patch_size * idx_1[0]:patch_size * (idx_1[0] + 1),
                            patch_size * idx_1[1]:patch_size * (idx_1[1] + 1)]

            patch_2 = image[patch_size * idx_2[0]:patch_size * (idx_2[0] + 1),
                            patch_size * idx_2[1]:patch_size * (idx_2[1] + 1)]

            exchanged_image[patch_size * idx_1[0]:patch_size * (idx_1[0] + 1),
                            patch_size * idx_1[1]:patch_size * (idx_1[1] + 1)] = patch_2

            exchanged_image[patch_size * idx_2[0]:patch_size * (idx_2[0] + 1),
                            patch_size * idx_2[1]:patch_size * (idx_2[1] + 1)] = patch_1

            inconsistency = (labels[patch_size * idx_1[0]:patch_size * (idx_1[0] + 1),
                                  patch_size * idx_1[1]:patch_size * (idx_1[1] + 1)] !=
                             labels[patch_size * idx_2[0]:patch_size * (idx_2[0] + 1),
                                  patch_size * idx_2[1]:patch_size * (idx_2[1] + 1)]).astype(numpy.uint8)

            change_map[patch_size * idx_1[0]:patch_size * (idx_1[0] + 1),
                       patch_size * idx_1[1]:patch_size * (idx_1[1] + 1)] = inconsistency

            change_map[patch_size * idx_2[0]:patch_size * (idx_2[0] + 1),
                       patch_size * idx_2[1]:patch_size * (idx_2[1] + 1)] = inconsistency

        return exchanged_image, change_map

    def intER_image_patch_exchange(
        self,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        object_1: numpy.ndarray,
        object_2: numpy.ndarray,
        patch_sz: int
    ):
        """
        Performs inter-image patch exchange based on object-level clustering and patch size.

        Args:
            img_1 (torch.Tensor): The first input image.
            img_2 (torch.Tensor): The second input image.
            object_1 (numpy.ndarray): Object segmentation map for img_1.
            object_2 (numpy.ndarray): Object segmentation map for img_2.
            patch_sz (int): Size of the patch to exchange.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Exchanged image and corresponding change label.
        """
        # Combine images and object maps
        concat_img = numpy.concatenate([img_1, img_2], axis=1)
        object_2 = numpy.max(object_1) + 1 + object_2
        concat_object = numpy.concatenate([object_1, object_2], axis=1)

        # Number of unique objects
        obj_num = numpy.max(concat_object)

        # Feature vector calculation (mean and std for each object)
        feat_vect = numpy.zeros((obj_num + 1, 6))
        for obj_idx in range(obj_num + 1):
            obj_pixels = concat_img[concat_object == obj_idx]
            if obj_pixels.size > 0:
                feat_vect[obj_idx] = numpy.concatenate([numpy.mean(obj_pixels, axis=0), numpy.std(obj_pixels, axis=0)], axis=0)

        # DBSCAN clustering
        clustering = DBSCAN(eps=7.5, min_samples=10, n_jobs=1).fit(feat_vect)
        clustered_labels = clustering.labels_

        # Create a clustered map
        clustered_map = numpy.zeros(concat_img.shape[:2], dtype=numpy.uint8)
        for obj_idx in range(obj_num + 1):
            clustered_map[concat_object == obj_idx] = clustered_labels[obj_idx]

        # Separate labels for the two images
        label_1 = clustered_map[:, :img_1.shape[1]]
        label_2 = clustered_map[:, img_1.shape[1]:]

        # Identify change regions
        change_label = (label_1 != label_2).astype(numpy.uint8)

        # Determine number of patches in rows and columns
        patch_num_in_raw = img_1.shape[0] // patch_sz
        patch_idx = numpy.arange(patch_num_in_raw ** 2)
        numpy.random.shuffle(patch_idx)

        # Determine number of patches to exchange
        exchange_patch_num = int(self.exchange_ratio * (patch_num_in_raw ** 2))
        exchange_patch_idx = patch_idx[:exchange_patch_num]

        # Initialize output images
        exchange_img = img_1.copy()
        exchange_change_label = numpy.zeros(img_1.shape[:2], dtype=numpy.uint8)

        # Exchange patches between the two images
        for idx in exchange_patch_idx:
            patch_idx = numpy.unravel_index(idx, (patch_num_in_raw, patch_num_in_raw))
            row_start = patch_sz * patch_idx[0]
            col_start = patch_sz * patch_idx[1]

            # Perform patch exchange
            exchange_img[row_start:row_start + patch_sz, col_start:col_start + patch_sz] = \
                img_2[row_start:row_start + patch_sz, col_start:col_start + patch_sz]

            # Update change label for the exchanged region
            exchange_change_label[row_start:row_start + patch_sz, col_start:col_start + patch_sz] = \
                change_label[row_start:row_start + patch_sz, col_start:col_start + patch_sz]

        return exchange_img, exchange_change_label
