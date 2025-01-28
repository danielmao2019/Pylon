from typing import Tuple, Dict, Any
import random
import numpy as np
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
            pre_image = self.dataset[idx]['inputs']['image']
            class_labels = self._perform_clustering(pre_image)
            post_image, label = self._intra_image_patch_exchange(pre_image, class_labels, patch_size)
        else:
            img_1 = self.dataset[idx]['inputs']['image']
            img_2 = self.dataset[random.choice(range(len(self.dataset)))]['inputs']['image']
            objects_1 = self._segment_objects(img_1)
            objects_2 = self._segment_objects(img_2)
            post_image, label = self._inter_image_patch_exchange(img_1, img_2, objects_1, objects_2, patch_size)

        return pre_image, post_image, label

    def _segment_objects(self, image: torch.Tensor) -> np.ndarray:
        """
        Perform object segmentation using SLIC.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            np.ndarray: Object segmentation map.
        """
        return slic(image.numpy(), n_segments=self.n_segments, start_label=0)

    def _perform_clustering(self, image: torch.Tensor) -> np.ndarray:
        """
        Perform clustering on image segments using DBSCAN.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            np.ndarray: Clustered labels map.
        """
        segments = self._segment_objects(image)
        num_segments = np.max(segments)

        features = np.zeros((num_segments + 1, 6))
        for segment_idx in range(num_segments + 1):
            segment_mask = segments == segment_idx
            features[segment_idx] = np.concatenate([
                np.mean(image[segment_mask], axis=0),
                np.std(image[segment_mask], axis=0)
            ])

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean').fit(features)
        cluster_labels = clustering.labels_

        clustered_map = np.zeros(image.shape[:2], dtype=int)
        for segment_idx in range(num_segments + 1):
            clustered_map[segments == segment_idx] = cluster_labels[segment_idx] + 1

        return clustered_map

    def _intra_image_patch_exchange(self, image: torch.Tensor, labels: np.ndarray, patch_size: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Perform intra-image patch exchange.

        Args:
            image (torch.Tensor): Input image tensor.
            labels (np.ndarray): Clustering labels map.
            patch_size (int): Size of patches to exchange.

        Returns:
            Tuple[torch.Tensor, np.ndarray]: Modified image and change label map.
        """
        num_patches = image.shape[0] // patch_size
        patch_indices = np.arange(num_patches ** 2)
        np.random.shuffle(patch_indices)

        num_exchanges = int(num_patches ** 2 * self.exchange_ratio)
        exchanged_image = image.clone()
        change_map = np.zeros(image.shape[:2], dtype=np.uint8)

        for i in range(0, num_exchanges, 2):
            idx_1 = np.unravel_index(patch_indices[i], (num_patches, num_patches))
            idx_2 = np.unravel_index(patch_indices[i + 1], (num_patches, num_patches))

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
                                  patch_size * idx_2[1]:patch_size * (idx_2[1] + 1)]).astype(np.uint8)

            change_map[patch_size * idx_1[0]:patch_size * (idx_1[0] + 1),
                       patch_size * idx_1[1]:patch_size * (idx_1[1] + 1)] = inconsistency

            change_map[patch_size * idx_2[0]:patch_size * (idx_2[0] + 1),
                       patch_size * idx_2[1]:patch_size * (idx_2[1] + 1)] = inconsistency

        return exchanged_image, change_map

    def intER_image_patch_exchange(self, img_1: torch.Tensor, img_2: torch.Tensor, object_1: numpy.ndarray, object_2: numpy.ndarray, patch_sz):
        concat_img = numpy.concatenate([img_1, img_2], axis=1)

        object_2 = numpy.max(object_1) + 1 + object_2
        concat_object = numpy.concatenate([object_1, object_2], axis=1)

        obj_num = numpy.max(concat_object)

        feat_vect = numpy.zeros((obj_num + 1, 6))
        for obj_idx in range(0, obj_num + 1):
            feat_vect[obj_idx - 1] = numpy.concatenate(
                [numpy.mean(concat_img[concat_object == obj_idx], axis=0),
                 numpy.std(concat_img[concat_object == obj_idx], axis=0)], axis=0)

        # You need to tune these two parameters carefully
        clustering = DBSCAN(eps=7.5, min_samples=10, n_jobs=1).fit(feat_vect)
        clustered_labels = clustering.labels_

        clustered_map = numpy.zeros(concat_img.shape[0:2]).astype(numpy.uint8)
        for obj_idx in range(0, obj_num + 1):
            clustered_map[concat_object == obj_idx] = clustered_labels[obj_idx]

        label_1 = clustered_map[:, 0:img_1.shape[1]]
        label_2 = clustered_map[:, img_1.shape[1]:]

        change_label = (label_1 != label_2).astype(numpy.uint8)
        # change_label[label_1 == -1] = 255
        # change_label[label_2 == -1] = 255

        patch_num_in_raw = img_1.shape[0] // patch_sz
        patch_idx = numpy.arange(patch_num_in_raw ** 2)
        numpy.random.shuffle(patch_idx)

        exchange_patch_num = int(self.exchange_ratio * (patch_num_in_raw ** 2))
        exchange_patch_idx = patch_idx[0:exchange_patch_num]
        exchange_img = img_1.copy()
        exchange_change_label = numpy.zeros(img_1.shape[0:2]).astype(numpy.uint8)

        for idx in range(0, exchange_patch_num):
            patch_idx = numpy.unravel_index(exchange_patch_idx[idx], (patch_num_in_raw, patch_num_in_raw))

            exchange_img[patch_sz * patch_idx[0]: patch_sz * (patch_idx[0] + 1),
            patch_sz * patch_idx[1]: patch_sz * (patch_idx[1] + 1)] = \
                img_2[patch_sz * patch_idx[0]: patch_sz * (patch_idx[0] + 1),
                patch_sz * patch_idx[1]: patch_sz * (patch_idx[1] + 1)]

            exchange_change_label[patch_sz * patch_idx[0]: patch_sz * (patch_idx[0] + 1),
            patch_sz * patch_idx[1]: patch_sz * (patch_idx[1] + 1)] = \
                change_label[patch_sz * patch_idx[0]: patch_sz * (patch_idx[0] + 1),
                patch_sz * patch_idx[1]: patch_sz * (patch_idx[1] + 1)]

        return exchange_img, exchange_change_label
