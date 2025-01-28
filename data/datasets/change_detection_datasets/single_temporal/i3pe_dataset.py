from typing import Tuple, Dict, Any
import random
import numpy
import torch
from sklearn.cluster import DBSCAN
from skimage.segmentation import slic
from data.datasets import BaseSyntheticDataset


class I3PEDataset(BaseSyntheticDataset):

    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    n_segments = 1000
    eps = 7
    min_samples = 10
    scale_factor = [16, 32, 64, 128]

    def __init__(self, exchange_ratio, **kwargs) -> None:
        self.exchange_ratio = exchange_ratio
        super(I3PEDataset, self).__init__(**kwargs)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        exchange_type_seed = random.random()
        patch_sz = random.choice(self.scale_factor)

        if exchange_type_seed < 0.5:  # Performing Intra-Image Patch Exchange Method
            pre_img = self.dataset[idx]['inputs']['image']
            class_label = self._get_clustering(pre_img)
            post_img, label = self.intRA_image_patch_exchange(pre_img, class_label, patch_sz=patch_sz)

        else:  # Performing Inter-Image Patch Exchange Method
            img_1 = self.dataset[idx]['inputs']['image']
            img_2 = self.dataset[random.choice(range(len(self.dataset)))]['inputs']['image']
            img_1_objects = self._get_objects(img_1)
            img_2_objects = self._get_objects(img_2)
            post_img, label = self.intER_image_patch_exchange(
                img_1, img_2, img_1_objects, img_2_objects, patch_sz=patch_sz,
            )

        return pre_img, post_img, label

    def _get_objects(self, image: torch.Tensor) -> numpy.ndarray:
        return slic(image.numpy(), n_segments=self.n_segments, start_label=0)

    def _get_clustering(self, image: torch.Tensor) -> numpy.ndarray:
        objects = self._get_objects(image)
        obj_num = numpy.max(objects)
        feat_vect = numpy.zeros(size=(obj_num + 1, 6))
        for obj_idx in range(0, obj_num + 1):
            feat_vect[obj_idx] = numpy.concatenate([
                numpy.mean(image[objects == obj_idx], axis=0),
                numpy.std(image[objects == obj_idx], axis=0),
            ], axis=0)
        clustered_labels = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, leaf_size=50, n_jobs=12, metric='euclidean'
        ).fit(feat_vect).labels_
        clustered_map = numpy.zeros(size=image.shape[0:2])
        for obj_idx in range(obj_num + 1):
            clustered_map[objects == obj_idx] = clustered_labels[obj_idx]
        clustered_map = clustered_map + 1
        return clustered_map

    def intRA_image_patch_exchange(self, img: torch.Tensor, class_label: numpy.ndarray, patch_sz):
        patch_num_in_raw = img.shape[0] // patch_sz
        patch_idx = numpy.arange(patch_num_in_raw ** 2)
        numpy.random.shuffle(patch_idx)

        exchange_patch_num = int((patch_num_in_raw ** 2) * self.exchange_ratio)

        exchange_img = img.copy()
        change_label = numpy.zeros(img.shape[0:2]).astype(numpy.uint8)

        for i in range(0, exchange_patch_num, 2):
            first_patch_idx = numpy.unravel_index(patch_idx[i], (patch_num_in_raw, patch_num_in_raw))
            second_patch_idx = numpy.unravel_index(patch_idx[i + 1], (patch_num_in_raw, patch_num_in_raw))

            first_patch = img[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
                          patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)]
            second_patch = img[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
                           patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)]

            temp = first_patch.copy()
            exchange_img[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
            patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)] = second_patch
            exchange_img[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
            patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)] = temp

            incons_label = \
                (class_label[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
                 patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)] !=
                 class_label[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
                 patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)]).astype(numpy.uint8)

            #
            change_label[patch_sz * first_patch_idx[0]: patch_sz * (first_patch_idx[0] + 1),
            patch_sz * first_patch_idx[1]: patch_sz * (first_patch_idx[1] + 1)] = incons_label

            change_label[patch_sz * second_patch_idx[0]: patch_sz * (second_patch_idx[0] + 1),
            patch_sz * second_patch_idx[1]: patch_sz * (second_patch_idx[1] + 1)] = incons_label

        return exchange_img, change_label

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
