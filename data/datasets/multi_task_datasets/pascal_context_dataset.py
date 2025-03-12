from typing import Tuple, List, Dict, Any, Optional
import os
import json
import scipy
import numpy
import torch
from data.datasets import BaseDataset
import utils


class PASCALContextDataset(BaseDataset):
    __doc__ = r"""
    PASCAL Context dataset for multi-task learning with semantic segmentation, human part segmentation, surface normal estimation, and saliency detection tasks.
    
    For detailed documentation, see: docs/datasets/multi_task/pascal_context.md
    """

    SPLIT_OPTIONS = ['train', 'val']
    DATASET_SIZE = {
        'train': 4998,
        'val': 5105,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['semantic_segmentation', 'parts_target', 'parts_inst_mask', 'normal_estimation', 'saliency_estimation']
    SHA1SUM = "5cd337198ead0768975610a135e26257153198c7"

    HUMAN_PART = {1: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 1,
                      'lhand': 1, 'llarm': 1, 'llleg': 1, 'luarm': 1, 'luleg': 1, 'mouth': 1,
                      'neck': 1, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 1,
                      'rhand': 1, 'rlarm': 1, 'rlleg': 1, 'ruarm': 1, 'ruleg': 1, 'torso': 1},
                  4: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 4,
                      'lhand': 3, 'llarm': 3, 'llleg': 4, 'luarm': 3, 'luleg': 4, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 4,
                      'rhand': 3, 'rlarm': 3, 'rlleg': 4, 'ruarm': 3, 'ruleg': 4, 'torso': 2},
                  6: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 6,
                      'lhand': 4, 'llarm': 4, 'llleg': 6, 'luarm': 3, 'luleg': 5, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 6,
                      'rhand': 4, 'rlarm': 4, 'rlleg': 6, 'ruarm': 3, 'ruleg': 5, 'torso': 2},
                  14: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 14,
                       'lhand': 8, 'llarm': 7, 'llleg': 13, 'luarm': 6, 'luleg': 12, 'mouth': 1,
                       'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 11,
                       'rhand': 5, 'rlarm': 4, 'rlleg': 10, 'ruarm': 3, 'ruleg': 9, 'torso': 2}
                  }

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    CONTEXT_CATEGORY_LABELS = [0,
                               2, 23, 25, 31, 34,
                               45, 59, 65, 72, 98,
                               397, 113, 207, 258, 284,
                               308, 347, 368, 416, 427]

    HUMAN_PARTS_CATEGORY = 15

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def __init__(
        self,
        num_human_parts: Optional[int] = 6,
        area_thres: Optional[int] = 0,
        **kwargs,
    ) -> None:
        self.num_human_parts = num_human_parts
        self.area_thres = area_thres
        super(PASCALContextDataset, self).__init__(**kwargs)

    def _init_annotations(self) -> None:
        with open(os.path.join(os.path.join(self.data_root, 'ImageSets', 'Context', f"{self.split}.txt")), mode="r") as f:
            lines = f.read().splitlines()
        ids = []
        image_paths = []
        semantic_paths = []
        parts_paths = []
        normal_paths = []
        saliency_paths = []
        for line in lines:
            ids.append(line.rstrip('\n'))
            # Image
            image_path = os.path.join(self.data_root, "JPEGImages", line + ".jpg")
            assert os.path.isfile(image_path)
            image_paths.append(image_path)
            # Semantic Segmentation
            semantic_path = self._get_semseg_fname(line)
            assert os.path.isfile(semantic_path)
            semantic_paths.append(semantic_path)
            # Human Parts
            parts_path = os.path.join(self.data_root, "human_parts", line + ".mat")
            assert os.path.isfile(parts_path)
            parts_paths.append(parts_path)
            # Normal estimation
            normal_path = os.path.join(self.data_root, "normals_distill", line + ".png")
            assert os.path.isfile(normal_path)
            normal_paths.append(normal_path)
            # Saliency estimation
            saliency_path = os.path.join(self.data_root, "sal_distill", line + ".png")
            assert os.path.isfile(saliency_path)
            saliency_paths.append(saliency_path)
        assert len(ids) == len(image_paths)
        assert len(ids) == len(semantic_paths)
        assert len(ids) == len(parts_paths)
        assert len(ids) == len(normal_paths)
        assert len(ids) == len(saliency_paths)
        # construct annotations
        self.annotations: List[Dict[str, str]] = [{
            'id': ids[idx],
            'image': image_paths[idx],
            'semantic': semantic_paths[idx],
            'parts': parts_paths[idx],
            'normal': normal_paths[idx],
            'saliency': saliency_paths[idx],
        } for idx in range(len(image_paths))]
        self._init_parts()
        self._init_normal()

    def _init_parts(self) -> None:
        self.cat_part = json.load(open(os.path.join(self.data_root, "db_info", "pascal_part.json"), 'r'))
        self.cat_part["15"] = self.HUMAN_PART[self.num_human_parts]
        self.parts_file = os.path.join(self.data_root, 'ImageSets', 'Parts', f"{self.split}.txt")

        print("Initializing dataloader for PASCAL {} set".format(''.join(self.split)))
        if not self._check_preprocess_parts():
            print('Pre-processing PASCAL dataset for human parts, this will take long, but will be done only once.')
            self._preprocess_parts()

    def _check_preprocess_parts(self):
        _obj_list_file = self.parts_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.part_obj_dict = json.load(open(_obj_list_file, 'r'))

            return list(numpy.sort([str(x) for x in self.part_obj_dict.keys()])) == list(numpy.sort([ann['id'] for ann in self.annotations]))

    def _preprocess_parts(self):
        self.part_obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.annotations)):
            # Read object masks and get number of objects
            if ii % 100 == 0:
                print("Processing image: {}".format(ii))
            part_mat = scipy.io.loadmat(
                os.path.join(self.data_root, 'human_parts', '{}.mat'.format(self.annotations[ii]['id'])))
            n_obj = len(part_mat['anno'][0][0][1][0])

            # Get the categories from these objects
            _cat_ids = []
            for jj in range(n_obj):
                obj_area = numpy.sum(part_mat['anno'][0][0][1][0][jj][2])
                obj_cat = int(part_mat['anno'][0][0][1][0][jj][1])
                if obj_area > self.area_thres:
                    _cat_ids.append(int(part_mat['anno'][0][0][1][0][jj][1]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.part_obj_dict[self.annotations[ii]['id']] = _cat_ids

        with open(self.parts_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.annotations[0]['id'], json.dumps(self.part_obj_dict[self.annotations[0]['id']])))
            for ii in range(1, len(self.annotations)):
                outfile.write(
                    ',\n\t"{:s}": {:s}'.format(self.annotations[ii]['id'], json.dumps(self.part_obj_dict[self.annotations[ii]['id']])))
            outfile.write('\n}\n')
        return

    def _init_normal(self) -> None:
        with open(os.path.join(self.data_root, "db_info", "nyu_classes.json")) as f:
            cls_nyu = json.load(f)
        with open(os.path.join(self.data_root, "db_info", "context_classes.json")) as f:
            cls_context = json.load(f)

        self.normals_valid_classes = []
        for cl_nyu in cls_nyu:
            if cl_nyu in cls_context and cl_nyu != 'unknown':
                self.normals_valid_classes.append(cls_context[cl_nyu])

        # Custom additions due to incompatibilities
        self.normals_valid_classes.append(cls_context['tvmonitor'])

    # ====================================================================================================
    # load methods
    # ====================================================================================================

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = self._load_image(idx)
        labels = {}
        labels.update(self._load_semantic(idx))
        labels.update(self._load_parts(idx))
        labels.update(self._load_normal(idx))
        labels.update(self._load_saliency(idx))
        meta_info = {
            'id': self.annotations[idx]['id'],
            'image_resolution': inputs['image'].shape,
        }
        return inputs, labels, meta_info

    def _load_image(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'image': utils.io.load_image(
            filepath=self.annotations[idx]['image'],
            dtype=torch.float32, sub=None, div=255.,
        )}

    def _load_semantic(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'semantic_segmentation': utils.io.load_image(
            filepath=self.annotations[idx]['semantic'],
            dtype=torch.int64, sub=None, div=None,
        )}

    def _load_parts(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.HUMAN_PARTS_CATEGORY not in self.part_obj_dict[self.annotations[idx]['id']]:
            return {
                'parts_target': None,
                'parts_inst_mask': None,
            }
        parts_mat = scipy.io.loadmat(self.annotations[idx]['parts'])['anno'][0][0][1][0]
        inst_mask = target = None
        num_obj = len(parts_mat)
        for obj_idx in range(num_obj):
            has_human = parts_mat[obj_idx][1][0][0] == self.HUMAN_PARTS_CATEGORY
            has_parts = len(parts_mat[obj_idx][3]) != 0
            if has_human and has_parts:
                if inst_mask is None:
                    inst_mask = torch.from_numpy(parts_mat[obj_idx][2].astype(numpy.float32))
                    target = torch.zeros(size=inst_mask.shape, dtype=torch.float32)
                else:
                    new_inst_mask = torch.from_numpy(parts_mat[obj_idx][2].astype(numpy.float32))
                    inst_mask = torch.maximum(inst_mask, new_inst_mask)

                n_parts = len(parts_mat[obj_idx][3][0])
                for part_i in range(n_parts):
                    cat_part = str(parts_mat[obj_idx][3][0][part_i][0][0])
                    mask_id = self.cat_part[str(self.HUMAN_PARTS_CATEGORY)][cat_part]
                    mask = parts_mat[obj_idx][3][0][part_i][1].astype(bool)
                    target[mask] = mask_id
        assert target is not None
        assert inst_mask is not None
        return {
            'parts_target': target,
            'parts_inst_mask': inst_mask,
        }

    def _load_normal(self, idx: int) -> Dict[str, torch.Tensor]:
        _tmp = utils.io.load_image(
            filepath=self.annotations[idx]['normal'],
            dtype=torch.float32, sub=None, div=255.,
        )
        _tmp = _tmp * 2 - 1

        labels = scipy.io.loadmat(os.path.join(self.data_root, 'pascal-context', 'trainval', self.annotations[idx]['id'] + '.mat'))
        labels = torch.from_numpy(labels['LabelMap'].astype(numpy.int64))

        normal = torch.zeros(_tmp.shape, dtype=torch.float32)
        for x in torch.unique(labels):
            if x in self.normals_valid_classes:
                normal[:, labels == x] = _tmp[:, labels == x]

        return {'normal_estimation': normal}

    def _load_saliency(self, idx: int) -> Dict[str, torch.Tensor]:
        saliency = utils.io.load_image(
            filepath=self.annotations[idx]['saliency'],
            dtype=torch.float32, sub=None, div=255.,
        )
        saliency = (saliency > 0.5).type(torch.float32)
        return {'saliency_estimation': saliency}

    # ====================================================================================================
    # helpers
    # ====================================================================================================

    def _get_semseg_fname(self, fname):
        fname_voc = os.path.join(self.data_root, 'semseg', 'VOC12', fname + '.png')
        fname_context = os.path.join(self.data_root, 'semseg', 'pascal-context', fname + '.png')
        if os.path.isfile(fname_voc):
            seg = fname_voc
        elif os.path.isfile(fname_context):
            seg = fname_context
        else:
            raise ValueError('Segmentation for im: {} was not found'.format(fname))
        return seg
