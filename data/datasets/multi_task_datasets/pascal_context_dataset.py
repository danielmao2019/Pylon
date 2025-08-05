from typing import Tuple, List, Dict, Any, Optional
import os
import json
import scipy
import numpy
import torch
from data.datasets.multi_task_datasets.base_multi_task_dataset import BaseMultiTaskDataset
import utils


class PASCALContextDataset(BaseMultiTaskDataset):
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

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        # PASCAL Context has parameters that affect which tasks are loaded
        version_dict.update({
            'num_human_parts': self.num_human_parts,
            'area_thres': self.area_thres,
        })
        return version_dict

    # ====================================================================================================
    # load methods
    # ====================================================================================================

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = self._load_image(idx)
        labels = {}
        
        # Only load selected labels to optimize disk I/O
        if 'semantic_segmentation' in self.selected_labels:
            labels.update(self._load_semantic(idx))
        if any(label in self.selected_labels for label in ['parts_target', 'parts_inst_mask']):
            parts_labels = self._load_parts(idx)
            for key, value in parts_labels.items():
                if key in self.selected_labels:
                    labels[key] = value
        if 'normal_estimation' in self.selected_labels:
            labels.update(self._load_normal(idx))
        if 'saliency_estimation' in self.selected_labels:
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

    def display_datapoint(
        self,
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> 'html.Div':
        """Display PASCAL Context multi-task datapoint with all modalities.
        
        This method visualizes PASCAL Context tasks: RGB image, semantic segmentation,
        human parts segmentation, surface normal estimation, and saliency detection.
        
        Args:
            datapoint: Dictionary containing inputs, labels, and meta_info
            class_labels: Optional mapping from class indices to label names
            camera_state: Optional camera state (unused for 2D displays)
            settings_3d: Optional 3D settings (unused for 2D displays)
            
        Returns:
            HTML div containing the multi-task visualization
            
        Raises:
            AssertionError: If datapoint structure is invalid
        """
        from dash import html
        from data.viewer.utils.atomic_displays import (
            create_image_display,
            create_segmentation_display,
            create_normal_display,
            get_image_display_stats,
            get_segmentation_display_stats,
            get_normal_display_stats
        )
        from data.viewer.utils.display_utils import (
            ParallelFigureCreator,
            create_figure_grid,
            create_standard_datapoint_layout,
            create_statistics_display
        )
        
        # CRITICAL: Input validation with fail-fast assertions
        assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
        assert 'inputs' in datapoint, f"datapoint missing 'inputs', got keys: {list(datapoint.keys())}"
        assert 'labels' in datapoint, f"datapoint missing 'labels', got keys: {list(datapoint.keys())}"
        
        inputs = datapoint['inputs']
        labels = datapoint['labels']
        
        assert isinstance(inputs, dict), f"inputs must be dict, got {type(inputs)}"
        assert isinstance(labels, dict), f"labels must be dict, got {type(labels)}"
        
        # Validate expected PASCAL Context data keys
        assert 'image' in inputs, f"inputs missing 'image', got keys: {list(inputs.keys())}"
        assert 'semantic_segmentation' in labels, f"labels missing 'semantic_segmentation', got keys: {list(labels.keys())}"
        assert 'normal_estimation' in labels, f"labels missing 'normal_estimation', got keys: {list(labels.keys())}"
        assert 'saliency_estimation' in labels, f"labels missing 'saliency_estimation', got keys: {list(labels.keys())}"
        
        # Create figure tasks for parallel execution
        # Skip parts segmentation if None (human parts may not be present in all images)
        figure_tasks = [
            lambda: create_image_display(
                image=inputs['image'],
                title="RGB Image"
            ),
            lambda: create_segmentation_display(
                segmentation=labels['semantic_segmentation'],
                title="Semantic Segmentation",
                class_labels=class_labels
            ),
            lambda: create_normal_display(
                normal=labels['normal_estimation'],
                title="Surface Normal Estimation"
            ),
            lambda: create_segmentation_display(
                segmentation=labels['saliency_estimation'],
                title="Saliency Detection"
            )
        ]
        
        # Add parts segmentation if present
        if labels.get('parts_target') is not None:
            figure_tasks.append(
                lambda: create_segmentation_display(
                    segmentation=labels['parts_target'],
                    title="Human Parts Segmentation"
                )
            )
        
        # Create figures in parallel for better performance  
        max_workers = len(figure_tasks)
        figure_creator = ParallelFigureCreator(max_workers=max_workers, enable_timing=False)
        figures = figure_creator.create_figures_parallel(figure_tasks)
        
        # Create grid layout - adjust width based on number of figures
        if len(figures) == 5:
            width_style = "20%"  # 5 figures in a row
        else:
            width_style = "25%"  # 4 figures in a row
            
        figure_components = create_figure_grid(
            figures=figures,
            width_style=width_style,
            height_style="350px"
        )
        
        # Create statistics for each modality
        stats_data = [
            get_image_display_stats(inputs['image']),
            get_segmentation_display_stats(labels['semantic_segmentation']),
            get_normal_display_stats(labels['normal_estimation']),
            get_segmentation_display_stats(labels['saliency_estimation'])
        ]
        
        stats_titles = [
            "RGB Image Statistics",
            "Semantic Segmentation Statistics",
            "Surface Normal Statistics",
            "Saliency Detection Statistics"
        ]
        
        # Add parts statistics if present
        if labels.get('parts_target') is not None:
            stats_data.append(get_segmentation_display_stats(labels['parts_target']))
            stats_titles.append("Human Parts Statistics")
        
        # Adjust stats width based on number of stats
        if len(stats_data) == 5:
            stats_width = "20%"
        else:
            stats_width = "25%"
            
        stats_components = create_statistics_display(
            stats_data=stats_data,
            titles=stats_titles,
            width_style=stats_width
        )
        
        # Use standard layout with all components
        return create_standard_datapoint_layout(
            figure_components=figure_components,
            stats_components=stats_components,
            meta_info=datapoint.get('meta_info', {}),
            debug_outputs=datapoint.get('debug')
        )
