from typing import Any, Dict, List, Tuple, Optional
import os
import glob
import numpy as np
import torch
import json
import jsbeautifier
import open3d as o3d
from data.datasets.base_dataset import BaseDataset


def make_open3d_point_cloud(xyz, color=None):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) == 3:
            color = np.repeat(np.array(color)[np.newaxis, ...], xyz.shape[0], axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


class KITTIDataset(BaseDataset):
    """KITTI dataset for point cloud registration.

    Paper:
        Are we ready for autonomous driving? The KITTI vision benchmark suite
        https://ieeexplore.ieee.org/document/6248074
    """

    SPLIT_OPTIONS = ['train', 'val', 'test']
    SEQUENCES_SPLIT = {
        'train': ['00', '01', '02', '03', '04', '05'],
        'val': ['06', '07'],
        'test': ['08', '09', '10'],
    }
    DATASET_SIZE = {
        'train': 1358,
        'val': 180,
        'test': 555,
    }
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']

    def __init__(self, **kwargs) -> None:
        assert 'data_root' in kwargs, "data_root must be provided"
        self.annotations_cache_root = os.path.join(kwargs['data_root'], 'annotations_cache')
        os.makedirs(self.annotations_cache_root, exist_ok=True)
        self.icp_cache_root = os.path.join(kwargs['data_root'], 'icp_cache')
        os.makedirs(self.icp_cache_root, exist_ok=True)
        self.seq_pose_cache = {}
        super(KITTIDataset, self).__init__(**kwargs)

    def _init_annotations(self) -> None:
        if os.path.isfile(os.path.join(self.annotations_cache_root, f'{self.split}.json')):
            with open(os.path.join(self.annotations_cache_root, f'{self.split}.json'), 'r') as f:
                self.annotations = json.load(f)
            return
        self.annotations: List[Dict[str, Any]] = []
        sequences = self.SEQUENCES_SPLIT[self.split]
        for seq in sequences:
            seq_path = os.path.join(self.data_root, 'sequences', seq, "velodyne")
            fnames = sorted(glob.glob(os.path.join(seq_path, "*.bin")))
            inames = sorted(list(map(lambda x: int(os.path.split(x)[-1][:-4]), fnames)))
            all_odo = self.get_video_odometry(seq)
            all_pos = self.odometry_to_positions(all_odo)
            assert len(fnames) == len(inames) == len(all_odo) == len(all_pos), \
                f"{len(fnames)=}, {len(inames)=}, {len(all_odo)=}, {len(all_pos)=}"
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time_arr = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time_arr) > 0:
                    next_time = next_time_arr[0].item() + curr_time - 1
                    assert next_time in inames
                    self.annotations.append({
                        'seq': seq,
                        't0': curr_time,
                        't1': next_time,
                    })
                    curr_time = next_time + 1
                else:
                    curr_time += 1
        if self.split == 'test':
            self.annotations.remove({'seq': '08', 't0': 15, 't1': 58})
        with open(os.path.join(self.annotations_cache_root, f'{self.split}.json'), 'w') as f:
            f.write(jsbeautifier.beautify(json.dumps(self.annotations), jsbeautifier.default_options()))

    def get_video_odometry(self, seq: str, indices: Optional[List[int]] = None) -> np.ndarray:
        data_path = os.path.join(self.data_root, 'poses', f'{seq}.txt')
        if data_path not in self.seq_pose_cache:
            self.seq_pose_cache[data_path] = np.genfromtxt(data_path)
        if indices is None:
            return self.seq_pose_cache[data_path]
        else:
            return self.seq_pose_cache[data_path][indices]

    def odometry_to_positions(self, odometry: np.ndarray) -> np.ndarray:
        assert isinstance(odometry, np.ndarray), f"odometry is not a numpy array"
        assert odometry.ndim == 2 and odometry.shape[-1] == 12, f"{odometry.shape=}"
        num_odo = odometry.shape[0]
        T_w_cam0 = odometry.reshape(num_odo, 3, 4)
        homo_row = np.zeros((num_odo, 1, 4))
        homo_row[:, 0, 3] = 1
        T_w_cam0 = np.concatenate((T_w_cam0, homo_row), axis=1)
        assert T_w_cam0.shape == (num_odo, 4, 4), f"{T_w_cam0.shape=}"
        return T_w_cam0

    def apply_transform(self, pts: np.ndarray, trans: np.ndarray) -> np.ndarray:
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self) -> np.ndarray:
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        ann = self.annotations[idx]
        seq = ann['seq']
        t0 = ann['t0']
        t1 = ann['t1']
        two_odo = self.get_video_odometry(seq, indices=[t0, t1])
        two_pos = self.odometry_to_positions(two_odo)
        fname0 = os.path.join(self.data_root, 'sequences', seq, 'velodyne', f'{t0:06d}.bin')
        fname1 = os.path.join(self.data_root, 'sequences', seq, 'velodyne', f'{t1:06d}.bin')
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = f'{seq}_{t0}_{t1}'
        icp_cache_file = os.path.join(self.icp_cache_root, f'{key}.npy')
        if not os.path.exists(icp_cache_file):
            M = (self.velo2cam @ two_pos[0].T @ np.linalg.inv(two_pos[1].T)
                 @ np.linalg.inv(self.velo2cam)).T
            xyz0_t = self.apply_transform(xyz0, M)
            pcd0 = make_open3d_point_cloud(xyz0_t, [0.5, 0.5, 0.5])
            pcd1 = make_open3d_point_cloud(xyz1, [0, 1, 0])
            reg = o3d.pipelines.registration.registration_icp(
                pcd0, pcd1, 0.20, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
            )
            pcd0.transform(reg.transformation)
            gt_transform = M @ reg.transformation
            np.save(icp_cache_file, gt_transform)
        else:
            gt_transform = np.load(icp_cache_file)

        inputs = {
            'src_pc': {
                'pos': torch.from_numpy(xyz0).float(),
                'reflectance': torch.from_numpy(xyzr0[:, 3]).float().unsqueeze(-1),
            },
            'tgt_pc': {
                'pos': torch.from_numpy(xyz1).float(),
                'reflectance': torch.from_numpy(xyzr1[:, 3]).float().unsqueeze(-1),
            },
        }
        labels = {
            'transform': torch.from_numpy(gt_transform).float(),
        }
        meta_info = {
            'seq': seq,
            't0': t0,
            't1': t1,
        }
        return inputs, labels, meta_info
