from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.pcr_datasets.threedmatch_dataset import ThreeDMatchDataset, ThreeDLoMatchDataset
from utils.builders.builder import build_from_config
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'src_pc', 'tgt_pc', 'correspondences'}, f"{inputs.keys()=}"

    # Validate source point cloud
    src_pc = inputs['src_pc']
    assert isinstance(src_pc, PointCloud), f"src_pc is not PointCloud: {type(src_pc)=}"
    src_pos = src_pc.xyz
    assert src_pos.ndim == 2, f"src_pc.xyz should be 2-dimensional: {src_pos.shape=}"
    assert src_pos.shape[1] == 3, f"src_pc.xyz should have 3 coordinates: {src_pos.shape=}"
    assert src_pos.dtype == torch.float32, f"src_pc.xyz dtype incorrect: {src_pos.dtype=}"

    assert hasattr(src_pc, 'feat'), "src_pc missing feat field"
    assert isinstance(src_pc.feat, torch.Tensor), f"src_pc.feat is not torch.Tensor: {type(src_pc.feat)=}"
    assert src_pc.feat.ndim == 2, f"src_pc.feat should be 2-dimensional: {src_pc.feat.shape=}"
    assert src_pc.feat.shape[1] == 1, f"src_pc.feat should have 1 feature: {src_pc.feat.shape=}"
    assert src_pc.feat.dtype == torch.float32, f"src_pc.feat dtype incorrect: {src_pc.feat.dtype=}"
    assert src_pos.shape[0] == src_pc.feat.shape[0], \
        f"src_pc positions and features should have same number of points: {src_pos.shape[0]=}, {src_pc.feat.shape[0]=}"

    # Validate target point cloud
    tgt_pc = inputs['tgt_pc']
    assert isinstance(tgt_pc, PointCloud), f"tgt_pc is not PointCloud: {type(tgt_pc)=}"
    tgt_pos = tgt_pc.xyz
    assert tgt_pos.ndim == 2, f"tgt_pc.xyz should be 2-dimensional: {tgt_pos.shape=}"
    assert tgt_pos.shape[1] == 3, f"tgt_pc.xyz should have 3 coordinates: {tgt_pos.shape=}"
    assert tgt_pos.dtype == torch.float32, f"tgt_pc.xyz dtype incorrect: {tgt_pos.dtype=}"

    assert hasattr(tgt_pc, 'feat'), "tgt_pc missing feat field"
    assert isinstance(tgt_pc.feat, torch.Tensor), f"tgt_pc.feat is not torch.Tensor: {type(tgt_pc.feat)=}"
    assert tgt_pc.feat.ndim == 2, f"tgt_pc.feat should be 2-dimensional: {tgt_pc.feat.shape=}"
    assert tgt_pc.feat.shape[1] == 1, f"tgt_pc.feat should have 1 feature: {tgt_pc.feat.shape=}"
    assert tgt_pc.feat.dtype == torch.float32, f"tgt_pc.feat dtype incorrect: {tgt_pc.feat.dtype=}"
    assert tgt_pos.shape[0] == tgt_pc.feat.shape[0], \
        f"tgt_pc positions and features should have same number of points: {tgt_pos.shape[0]=}, {tgt_pc.feat.shape[0]=}"

    # Validate correspondences
    correspondences = inputs['correspondences']
    assert isinstance(correspondences, torch.Tensor), f"correspondences is not torch.Tensor: {type(correspondences)=}"
    assert correspondences.ndim == 2, f"correspondences should be 2-dimensional: {correspondences.shape=}"
    assert correspondences.shape[1] == 2, f"correspondences should have 2 columns (src, tgt indices): {correspondences.shape=}"
    assert correspondences.dtype == torch.int64, f"correspondences dtype incorrect: {correspondences.dtype=}"

    # Check correspondence indices are valid
    if correspondences.shape[0] > 0:
        assert torch.all(correspondences[:, 0] >= 0) and torch.all(correspondences[:, 0] < src_pc.num_points), \
            "Invalid source indices in correspondences"
        assert torch.all(correspondences[:, 1] >= 0) and torch.all(correspondences[:, 1] < tgt_pc.num_points), \
            "Invalid target indices in correspondences"


def validate_labels(labels: Dict[str, Any], src_pc: PointCloud, tgt_pc: PointCloud, gt_overlap: float, matching_radius: float = 0.1) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == {'transform'}, f"{labels.keys()=}"
    assert isinstance(labels['transform'], torch.Tensor), f"transform is not torch.Tensor: {type(labels['transform'])=}"
    assert labels['transform'].shape == (4, 4), f"transform shape incorrect: {labels['transform'].shape=}"
    assert labels['transform'].dtype == torch.float32, f"transform dtype incorrect: {labels['transform'].dtype=}"

    # Check transform is valid
    expected_last_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=labels['transform'].device)
    assert torch.allclose(labels['transform'][3, :], expected_last_row), \
        "Transform last row should be [0, 0, 0, 1]"

    # Check rotation is orthogonal
    rotation = labels['transform'][:3, :3]
    rotation_transpose = rotation.transpose(0, 1)
    identity = torch.eye(3, dtype=torch.float32, device=rotation.device)
    assert torch.allclose(rotation @ rotation_transpose, identity, atol=1e-5), \
        "Rotation matrix is not orthogonal"

    # Recompute overlap and validate against stored overlap
    from data.structures.three_d.point_cloud.ops.set_ops.intersection import compute_registration_overlap

    gt_transform = labels['transform']

    # Use matching_radius as positive_radius for consistency with dataset
    positive_radius = matching_radius

    recomputed_overlap = compute_registration_overlap(
        ref_points=tgt_pc.xyz,
        src_points=src_pc.xyz,
        transform=gt_transform,
        positive_radius=positive_radius
    )

    # Test 1: Registration quality - overlap should be reasonable for 3DMatch
    # 3DMatch dataset contains pairs with overlap > 0.3, so this should hold
    assert recomputed_overlap > 0.25, \
        f"PCR relationship poor: overlap={recomputed_overlap:.4f} too low"

    # Test 2: Overlap consistency - recomputed should match stored (within tolerance)
    # Allow slightly larger tolerance for 3DMatch due to real-world data variability
    overlap_diff = abs(recomputed_overlap - gt_overlap)
    assert overlap_diff < 0.05, \
        f"Overlap inconsistency: stored={gt_overlap:.4f}, " \
        f"recomputed={recomputed_overlap:.4f}, diff={overlap_diff:.4f}"

    # Test 3: Overlap bounds check
    assert 0.0 <= recomputed_overlap <= 1.0, \
        f"Recomputed overlap out of bounds: {recomputed_overlap:.4f}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    # BaseDataset automatically adds 'idx'
    expected_keys = {'idx', 'src_path', 'tgt_path', 'scene_name', 'overlap', 'src_frame', 'tgt_frame'}
    assert meta_info.keys() == expected_keys, f"{meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"{meta_info['idx']=}, {datapoint_idx=}"
    assert isinstance(meta_info['src_path'], str), f"src_path is not str: {type(meta_info['src_path'])=}"
    assert isinstance(meta_info['tgt_path'], str), f"tgt_path is not str: {type(meta_info['tgt_path'])=}"
    assert isinstance(meta_info['scene_name'], str), f"scene_name is not str: {type(meta_info['scene_name'])=}"
    assert isinstance(meta_info['overlap'], float), f"overlap is not float: {type(meta_info['overlap'])=}"
    assert isinstance(meta_info['src_frame'], int), f"src_frame is not int: {type(meta_info['src_frame'])=}"
    assert isinstance(meta_info['tgt_frame'], int), f"tgt_frame is not int: {type(meta_info['tgt_frame'])=}"
    assert 0.0 <= meta_info['overlap'] <= 1.0, f"overlap should be between 0 and 1: {meta_info['overlap']=}"


@pytest.mark.parametrize('threedmatch_dataset_config', ['train', 'val', 'test'], indirect=True)
def test_threedmatch_dataset(threedmatch_dataset_config, max_samples, get_samples_to_test):
    dataset = build_from_config(threedmatch_dataset_config)
    """Test the structure and content of dataset outputs."""
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(
            labels=datapoint['labels'],
            src_pc=datapoint['inputs']['src_pc'],
            tgt_pc=datapoint['inputs']['tgt_pc'],
            gt_overlap=datapoint['meta_info']['overlap'],
            matching_radius=dataset.matching_radius
        )
        validate_meta_info(datapoint['meta_info'], idx)

    # Use command line --samples if provided, otherwise test first 5 samples
    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)


@pytest.mark.parametrize('threedlomatch_dataset_config', ['train', 'val', 'test'], indirect=True)
def test_threedlomatch_dataset(threedlomatch_dataset_config, max_samples, get_samples_to_test):
    lomatch_dataset = build_from_config(threedlomatch_dataset_config)
    """Test the structure and content of ThreeDLoMatchDataset outputs."""
    assert isinstance(lomatch_dataset, torch.utils.data.Dataset)
    assert len(lomatch_dataset) > 0, "Dataset should not be empty"

    def validate_datapoint(idx: int) -> None:
        datapoint = lomatch_dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(
            labels=datapoint['labels'],
            src_pc=datapoint['inputs']['src_pc'],
            tgt_pc=datapoint['inputs']['tgt_pc'],
            gt_overlap=datapoint['meta_info']['overlap'],
            matching_radius=lomatch_dataset.matching_radius
        )
        validate_meta_info(datapoint['meta_info'], idx)

    # Use command line --samples if provided, otherwise test first 5 samples
    num_samples = get_samples_to_test(len(lomatch_dataset), max_samples)
    indices = random.sample(range(len(lomatch_dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
