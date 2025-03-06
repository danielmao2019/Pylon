from typing import Tuple, Dict, Any
import os
import os.path as osp
import random
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data, Dataset
from sklearn.neighbors import KDTree
import csv
import pickle

from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.change_detection.pair import Pair, MultiScalePair
from torch_points3d.metrics.urb3DCD_tracker import Urb3DCDTracker

from matplotlib import cm

import utils


IGNORE_LABEL: int = -1

URB3DCD_NUM_CLASSES = 7
viridis = cm.get_cmap('viridis', URB3DCD_NUM_CLASSES)

INV_OBJECT_LABEL = {
    0: "unchanged",
    1: "newlyBuilt",
    2: "deconstructed",
    3: "newVegetation",
    4: "vegetationGrowUp",
    5: "vegetationRemoved",
    6: "mobileObjects"
}

#V2
OBJECT_COLOR = np.asarray(
    [
        [67, 1, 84],  # 'unchanged'
        [0, 183, 255],  # 'newlyBuilt'
        [0, 12, 235],  # 'deconstructed'
        [0, 217, 33],  # 'newVegetation'
        [255, 230, 0],  # 'vegetationGrowUp'
        [255, 140, 0],  # 'vegetationRemoved'
        [255, 0, 0],  # 'mobileObjects'
    ]
)
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}



class Urb3DSimul(Dataset):
    """
    Definition of Urb3DCD Dataset
    """

    def __init__(self, nameInPly="params", comp_norm = False ):
        super(Urb3DSimul, self).__init__(None, None, None)
        self.class_labels = OBJECT_LABEL
        self._ignore_label = IGNORE_LABEL
        self.nameInPly = nameInPly
        self._init_annotations()
        self.num_classes = URB3DCD_NUM_CLASSES

    def _init_annotations(self) -> None:
        filesPC0 = []
        filesPC1 = []
        globPath = os.scandir(self.data_root)
        for dir in globPath:
            if dir.is_dir():
                curDir = os.scandir(dir)
                for f in curDir:
                    if f.name == "pointCloud0.ply":
                        filesPC0.append(f.path)
                    elif f.name == "pointCloud1.ply":
                        filesPC1.append(f.path)
                curDir.close()
        globPath.close()
        self.annotations = [
            {'pc_0_filepath': filePC0, 'pc_1_filepath': filePC1}
            for filePC0, filePC1 in zip(filesPC0, filesPC1)
        ]

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        pc0, pc1, label = self.clouds_loader(idx, nameInPly=self.nameInPly)
        if (hasattr(pc0, "multiscale")):
            batch = MultiScalePair(pos=pc0, pos_target=pc1, y=label)
        else:
            batch = Pair(pos=pc0, pos_target=pc1, y=label)
            batch.normalise()
        inputs = {
            'pc_0': pc0,
            'pc_1': pc1,
        }
        labels = {
            'change_map': label,
        }
        meta_info = self.annotations[idx]
        return inputs, labels, meta_info

    def clouds_loader(self, idx: int, nameInPly = "params"):
        print("Loading " + self.filesPC1[idx])
        pc = utils.io.load_point_cloud(self.filesPC1[idx], nameInPly=nameInPly)
        pc1 = pc[:, :3]
        gt = pc[:, 3].long()  # Labels should be at the 4th column 0:X 1:Y 2:Z 3:LAbel
        pc0 = utils.io.load_point_cloud(self.filesPC0[idx], nameInPly=nameInPly)[:, :3]
        return pc0.type(torch.float), pc1.type(torch.float), gt


class Urb3DSimulSphere(Urb3DSimul):
    """Variation of Urb3DCD that allows random sphere sampling within an area
    during training and validation. Spheres have a 2m radius.
    If sample_per_epoch is not specified, spheres are placed on a 2m grid.
    """
    
    def __init__(self, sample_per_epoch=100, radius=2, fix_cyl=False, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = GridSampling3D(size=radius / 10.0)
        self.fix_cyl = fix_cyl
        super().__init__(*args, **kwargs)
        self._prepare_centers()

    def __len__(self):
        return self._sample_per_epoch if self._sample_per_epoch > 0 else self.grid_regular_centers.shape[0]

    def _prepare_centers(self):
        """Prepares centers based on whether random sampling or grid sampling is used."""
        if self._sample_per_epoch > 0:
            self._prepare_random_sampling()
        else:
            self._prepare_grid_sampling()

    def _prepare_random_sampling(self):
        """Prepares centers for random sphere sampling."""
        self._centres_for_sampling = []
        
        for i in range(len(self.filesPC0)):
            pair = self._load_save(i)
            dataPC1 = Data(pos=pair.pos_target, y=pair.y)
            low_res = self._grid_sphere_sampling(dataPC1)
            centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
            centres[:, :3] = low_res.pos
            centres[:, 3] = i
            centres[:, 4] = low_res.y
            self._centres_for_sampling.append(centres)
        
        self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
        labels, label_counts = np.unique(self._centres_for_sampling[:, -1].numpy(), return_counts=True)
        self._label_counts = np.sqrt(label_counts.mean() / label_counts)
        self._label_counts /= np.sum(self._label_counts)
        self._labels = labels
        self.weight_classes = torch.tensor(self._label_counts, dtype=torch.float)
        
        if self.fix_cyl:
            self._prepare_fixed_sampling()
    
    def _prepare_fixed_sampling(self):
        """Fixes the sampled cylinders for consistency across epochs."""
        np.random.seed(1)
        chosen_labels = np.random.choice(self._labels, p=self._label_counts, size=(self._sample_per_epoch, 1))
        unique_labels, label_counts = np.unique(chosen_labels, return_counts=True)
        
        self._centres_for_sampling_fixed = []
        for label, count in zip(unique_labels, label_counts):
            valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, -1] == label]
            selected_idx = np.random.randint(low=0, high=valid_centres.shape[0], size=(count,))
            self._centres_for_sampling_fixed.append(valid_centres[selected_idx])
        
        self._centres_for_sampling_fixed = torch.cat(self._centres_for_sampling_fixed, 0)
    
    def _prepare_grid_sampling(self):
        """Prepares centers for regular grid sphere sampling."""
        self.grid_regular_centers = []
        grid_sampling = GridSampling3D(size=self._radius / 2)
        
        for i in range(len(self.filesPC0)):
            pair = self._load_save(i)
            dataPC1 = Data(pos=pair.pos_target, y=pair.y)
            grid_sample_centers = grid_sampling(dataPC1.clone())
            centres = torch.empty((grid_sample_centers.pos.shape[0], 4), dtype=torch.float)
            centres[:, :3] = grid_sample_centers.pos
            centres[:, 3] = i
            self.grid_regular_centers.append(centres)
        
        self.grid_regular_centers = torch.cat(self.grid_regular_centers, 0)
    
    def _load_save(self, i):
        """Loads and processes point cloud pairs."""
        pc0, pc1, label = self.clouds_loader(i, nameInPly=self.nameInPly)
        pair = Pair(pos=pc0, pos_target=pc1, y=label)
        pair.KDTREE_KEY_PC0 = KDTree(np.asarray(pc0), leaf_size=10)
        pair.KDTREE_KEY_PC1 = KDTree(np.asarray(pc1), leaf_size=10)
        return pair


class Urb3DSimulCylinder(Urb3DSimulSphere):
    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_fixed_sample(idx) if self.fix_cyl else self._get_random()
        return self._get_regular_sample(idx)

    def _get_fixed_sample(self, idx):
        """Retrieves a fixed cylindrical sample, ensuring it meets normalization criteria."""
        while idx < self._centres_for_sampling_fixed.shape[0]:
            centre, area_sel = self._extract_centre_info(self._centres_for_sampling_fixed, idx)
            pair = self._load_save(area_sel)
            pair_cylinders = self._sample_cylinder(pair, centre, area_sel)

            if pair_cylinders and self._normalize_pair(pair_cylinders):
                return pair_cylinders
            
            idx += 1
        return None  # Return None if no valid pair is found

    def _get_regular_sample(self, idx):
        """Retrieves a regular cylindrical sample, ensuring it meets normalization criteria."""
        while idx < self.grid_regular_centers.shape[0]:
            centre, area_sel = self._extract_centre_info(self.grid_regular_centers, idx)
            pair = self._load_save(area_sel)
            pair_cylinders = self._sample_cylinder(pair, centre, area_sel, apply_transform=True)

            if pair_cylinders and self._normalize_pair(pair_cylinders):
                return pair_cylinders

            print('pair not correct')
            idx += 1
        return None  # Return None if no valid pair is found

    def _get_random(self):
        """Randomly selects a cylindrical sample with bias toward low-frequency classes."""
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]

        if valid_centres.shape[0] == 0:
            return None  # Return None if no valid centers exist for the chosen label

        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_sel = centre[3].int()
        pair = self._load_save(area_sel)
        pair_cyl = self._sample_cylinder(pair, centre[:3], area_sel, apply_transform=True)

        if pair_cyl:
            pair_cyl.normalise()
        return pair_cyl

    def _sample_cylinder(self, pair, centre, area_sel, apply_transform=False):
        """Applies cylindrical sampling and optional transformations."""
        cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)

        dataPC0 = Data(pos=pair.pos, idx=torch.arange(pair.pos.shape[0]))
        setattr(dataPC0, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC0)

        dataPC1 = Data(pos=pair.pos_target, y=pair.y, idx=torch.arange(pair.pos_target.shape[0]))
        setattr(dataPC1, CylinderSampling.KDTREE_KEY, pair.KDTREE_KEY_PC1)

        dataPC0_cyl = cylinder_sampler(dataPC0)
        dataPC1_cyl = cylinder_sampler(dataPC1)

        return Pair(pos=dataPC0_cyl.pos, pos_target=dataPC1_cyl.pos, y=dataPC1_cyl.y,
                    idx=dataPC0_cyl.idx, idx_target=dataPC1_cyl.idx, area=area_sel)

    def _extract_centre_info(self, centres, idx):
        """Extracts center position and area selection from the given array."""
        centre = centres[idx, :3]
        area_sel = centres[idx, 3].int()
        return centre, area_sel

    def _normalize_pair(self, pair_cylinders):
        """Attempts to normalize the pair, handling errors gracefully."""
        try:
            pair_cylinders.normalise()
            return True
        except Exception as e:
            print(f"Normalization failed: {e}")
            print(f"pos shape: {pair_cylinders.pos.shape}, pos_target shape: {pair_cylinders.pos_target.shape}")
            return False

    def _load_save(self, i):
        """Loads and constructs a Pair object with KDTree keys."""
        pc0, pc1, label = self.clouds_loader(i, nameInPly=self.nameInPly)
        pair = Pair(pos=pc0, pos_target=pc1, y=label)

        pair.KDTREE_KEY_PC0 = KDTree(np.asarray(pc0[:, :-1]), leaf_size=10)
        pair.KDTREE_KEY_PC1 = KDTree(np.asarray(pc1[:, :-1]), leaf_size=10)

        return pair


class Urb3DCDDataset(BaseSiameseDataset): #Urb3DCDDataset Urb3DSimulDataset
    """ Wrapper around Semantic Kitti that creates train and test datasets.
        Parameters
        ----------
        dataset_opt: omegaconf.DictConfig
            Config dictionary that should contain
                - root,
                - split,
                - transform,
                - pre_transform
                - process_workers
        """
    INV_OBJECT_LABEL = INV_OBJECT_LABEL
    FORWARD_CLASS = "forward.urb3DSimulPairCyl.ForwardUrb3DSimulDataset"

    def __init__(self, dataset_opt):
        # self.pre_transform = dataset_opt.get("pre_transforms", None)
        super().__init__(dataset_opt)
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.TTA = False
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir
        self.train_dataset = Urb3DSimulCylinder(
            filePaths=self.dataset_opt.dataTrainFile,
            split="train",
            radius=self.radius,
            sample_per_epoch=self.sample_per_epoch,
            DA=self.DA,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Train"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            nameInPly=self.dataset_opt.nameInPly,
            fix_cyl=self.dataset_opt.fix_cyl,
        )
        self.val_dataset = Urb3DSimulCylinder(
            filePaths=self.dataset_opt.dataValFile,
            split="val",
            radius=self.radius,
            sample_per_epoch= int(self.sample_per_epoch / 2),
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Val"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            nameInPly=self.dataset_opt.nameInPly,
            fix_cyl=self.dataset_opt.fix_cyl,
        )
        self.test_dataset = Urb3DSimulCylinder(
            filePaths=self.dataset_opt.dataTestFile,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            preprocessed_dir=osp.join(self.preprocessed_dir, "Test"),
            reload_preproc=self.dataset_opt.load_preprocessed,
            nameInPly=self.dataset_opt.nameInPly,
        )

    @property
    def train_data(self):
        if type(self.train_dataset) == list:
            return self.train_dataset[0]
        else:
            return self.train_dataset

    @property
    def val_data(self):
        if type(self.val_dataset) == list:
            return self.val_dataset[0]
        else:
            return self.val_dataset

    @property
    def test_data(self):
        if type(self.test_dataset) == list:
            return self.test_dataset[0]
        else:
            return self.test_dataset

    @staticmethod
    def to_ply(pos, label, file, color=OBJECT_COLOR):
        """ Allows to save Urb3DCD predictions to disk using Urb3DCD color scheme
            Parameters
            ----------
            pos : torch.Tensor
                tensor that contains the positions of the points
            label : torch.Tensor
                predicted label
            file : string
                Save location
            """
        to_ply(pos, label, file, color=color)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool, full_pc=False, full_res=False):
        """Factory method for the tracker
            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
        return Urb3DCDTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                                 full_pc=full_pc, full_res=full_res, ignore_label=IGNORE_LABEL)


################################### UTILS #######################################


def to_ply(pos, label, file, color = OBJECT_COLOR, sf = None):
    """ Allows to save Urb3DCD predictions to disk using Urb3DCD color scheme
       Parameters
       ----------
       pos : torch.Tensor
           tensor that contains the positions of the points
       label : torch.Tensor
           predicted label
       file : string
           Save location
    """
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    if max(label)<= color.shape[0]:
        colors = color[np.asarray(label)]
    else:
        colors = color[np.zeros(pos.shape[0], dtype=np.int)]
    if sf is None:
        ply_array = np.ones(
            pos.shape[0],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
                   ("green", "u1"), ("blue", "u1"), ("pred", "u2")]
        )
    else:
        ply_array = np.ones(
            pos.shape[0],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
                   ("green", "u1"), ("blue", "u1"), ("pred", "u2"), ("sf","f4")]
        )
        ply_array["sf"] = np.asarray(sf)
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    ply_array["pred"] = np.asarray(label)
    el = PlyElement.describe(ply_array, "params")
    PlyData([el], byte_order=">").write(file)
