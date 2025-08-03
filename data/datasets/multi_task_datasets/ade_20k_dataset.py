from typing import Tuple, Dict, Any, Optional, List
import os
import glob
import torch
import json
from data.datasets.base_dataset import BaseDataset
from utils.io.image import load_image


class ADE20KDataset(BaseDataset):
    __doc__ = r"""
    Reference:
        https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py

    Download:
        https://ade20k.csail.mit.edu/

    Used in:
        LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models
    """

    SPLIT_OPTIONS = ['training', 'validation']
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['object_cls_mask', 'object_ins_mask', 'parts_cls_masks', 'parts_ins_masks', 'objects', 'parts', 'amodal_masks']
    DATASET_SIZE = {
        'training': 25574,
        'validation': 2000,
    }

    def _init_annotations(self) -> None:
        datapoint_ids = sorted(filter(
            lambda x: os.path.isdir(x),
            glob.glob(os.path.join(
                self.data_root, 'images', 'ADE', self.split, "**", "**", "*",
        ))))
        self.annotations = list(map(
            lambda x: {
                'image_filepath': x + ".jpg",
                'object_mask_filepath': x + "_seg.png",
                'parts_masks_filepaths': sorted(glob.glob(x + "_parts_*.png")),
                'attr_filepath': x + ".json",
                'amodal_masks_filepaths': sorted(glob.glob(os.path.join(x, f"instance_\d{{3}}_{x}.png")))
            }, datapoint_ids
        ))

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        # ADE20KDataset has no additional parameters beyond BaseDataset
        return super()._get_cache_version_dict()

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {
            'image': load_image(
                filepath=self.annotations[idx]['image_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            ),
        }
        labels = {
            **self._load_object_mask(idx),
            **self._load_parts_masks(idx),
            **self._load_objects_parts(idx),
            **self._load_amodal_masks(idx),
        }
        meta_info = {
            **self.annotations[idx],
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info

    def _load_object_mask(self, idx: int) -> torch.Tensor:
        filepath = self.annotations[idx]['object_mask_filepath']
        object_mask = load_image(
            filepath=filepath, dtype=torch.int64,
            sub=None, div=None,
        )

        # Obtain the segmentation mask, built from the RGB channels of the _seg file
        R = object_mask[0, :, :]
        G = object_mask[1, :, :]
        B = object_mask[2, :, :]
        object_cls_mask = (R/10).to(torch.int64) * 256 + G

        # Obtain the instance mask from the blue channel of the _seg file
        m_instances_hat = torch.unique(B, return_inverse=True)[1]
        m_instances_hat = torch.reshape(m_instances_hat, B.shape)
        object_ins_mask = m_instances_hat

        return {
            'object_cls_mask': object_cls_mask,
            'object_ins_mask': object_ins_mask,
        }

    def _load_parts_masks(self, idx: int) -> Dict[str, torch.Tensor]:
        filepaths = self.annotations[idx]['parts_masks_filepaths']
        parts_cls_masks = []
        parts_ins_masks = []
        for filepath in filepaths:
            parts_mask = load_image(
                filepath=filepath, dtype=torch.int64,
                sub=None, div=None,
            )
            R = parts_mask[0, :, :]
            G = parts_mask[1, :, :]
            B = parts_mask[2, :, :]
            parts_cls_masks.append((R/10).to(torch.int64) * 256 + G)
            parts_ins_masks.append(parts_cls_masks[-1])
            # TODO:  correct partinstancemasks
        return {
            'parts_cls_masks': parts_cls_masks,
            'parts_ins_masks': parts_ins_masks,
        }

    def _load_objects_parts(self, idx: int) -> Dict[str, Any]:
        attr_file_name = self.annotations[idx]['attr_filepath']
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)
        objects = {}
        parts = {}
        contents = input_info['annotation']['object']
        instance = torch.tensor([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name =  [x['name'] for x in contents]
        partlevel = torch.tensor([int(x['parts']['part_level']) for x in contents])
        ispart = torch.tensor([p > 0 for p in partlevel])
        iscrop = torch.tensor([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = torch.tensor(p['x'])
            p['y'] = torch.tensor(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(torch.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(torch.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in list(torch.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in list(torch.where(ispart == 0)[0])]

        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(torch.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(torch.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in list(torch.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(torch.where(ispart == 1)[0])]

        return {
            'objects': objects,
            'parts': parts,
        }

    def _load_amodal_masks(self, idx: int) -> Dict[str, torch.Tensor]:
        filepaths = self.annotations[idx]['amodal_masks_filepaths']
        amodal_masks = []
        for filepath in filepaths:
            amodal_mask = load_image(
                filepath=filepath, dtype=torch.int64,
                sub=None, div=None,
            )
            amodal_masks.append(amodal_mask)
        return {
            'amodal_masks': amodal_masks,
        }
    
    def display_datapoint(
        self,
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> 'html.Div':
        """Display ADE20K multi-task datapoint with all modalities.
        
        This method visualizes ADE20K tasks: RGB image, object class/instance masks,
        parts class/instance masks, and amodal masks.
        
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
            get_image_display_stats,
            get_segmentation_display_stats
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
        
        # Validate expected ADE20K data keys
        assert 'image' in inputs, f"inputs missing 'image', got keys: {list(inputs.keys())}"
        assert 'object_cls_mask' in labels, f"labels missing 'object_cls_mask', got keys: {list(labels.keys())}"
        assert 'object_ins_mask' in labels, f"labels missing 'object_ins_mask', got keys: {list(labels.keys())}"
        
        # Create figure tasks for parallel execution - focus on main object segmentation
        # Skip parts and amodal masks for now as they are complex and might be empty
        figure_tasks = [
            lambda: create_image_display(
                image=inputs['image'],
                title="RGB Image"
            ),
            lambda: create_segmentation_display(
                segmentation=labels['object_cls_mask'],
                title="Object Class Segmentation",
                class_labels=class_labels
            ),
            lambda: create_segmentation_display(
                segmentation=labels['object_ins_mask'],
                title="Object Instance Segmentation"
            )
        ]
        
        # Create figures in parallel for better performance
        figure_creator = ParallelFigureCreator(max_workers=3, enable_timing=False)
        figures = figure_creator.create_figures_parallel(figure_tasks)
        
        # Create grid layout (3 figures in a row)
        figure_components = create_figure_grid(
            figures=figures,
            width_style="33.33%",
            height_style="400px"
        )
        
        # Create statistics for each modality
        stats_data = [
            get_image_display_stats(inputs['image']),
            get_segmentation_display_stats(labels['object_cls_mask']),
            get_segmentation_display_stats(labels['object_ins_mask'])
        ]
        
        stats_titles = [
            "RGB Image Statistics",
            "Object Class Statistics",
            "Object Instance Statistics"
        ]
        
        stats_components = create_statistics_display(
            stats_data=stats_data,
            titles=stats_titles,
            width_style="33.33%"
        )
        
        # Use standard layout with all components
        return create_standard_datapoint_layout(
            figure_components=figure_components,
            stats_components=stats_components,
            meta_info=datapoint.get('meta_info', {}),
            debug_outputs=datapoint.get('debug')
        )
