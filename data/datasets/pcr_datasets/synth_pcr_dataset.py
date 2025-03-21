
class SynthPCRDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 num_points=1024,
                 noise_type='clean',
                 rot_mag=45.0,
                 trans_mag=0.5,
                 partial_p_keep=[0.7, 0.7],
                 ):
        """Synthetic Point Cloud Registration Dataset
        
        Args:
            root (str): Root directory containing PLY files
            split (str): Dataset split ('train' or 'test')
            num_points (int): Number of points to sample from each point cloud
            noise_type (str): Type of noise to apply ('clean', 'jitter', 'crop')
            rot_mag (float): Maximum rotation angle in degrees
            trans_mag (float): Maximum translation magnitude
            partial_p_keep (list): Proportion of points to keep for cropping [src_p, ref_p]
        """
        self.root = root
        self.split = split
        self.num_points = num_points
        self.noise_type = noise_type
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.partial_p_keep = partial_p_keep

        # Load all PLY files from the root directory
        self.ply_files = []
        for file in os.listdir(root):
            if file.endswith('.ply'):
                self.ply_files.append(os.path.join(root, file))
        
        # Split into train/test if needed
        if split in ['train', 'test']:
            np.random.seed(42)  # For reproducibility
            indices = np.random.permutation(len(self.ply_files))
            split_idx = int(0.8 * len(self.ply_files))  # 80% train, 20% test
            if split == 'train':
                self.ply_files = [self.ply_files[i] for i in indices[:split_idx]]
            else:
                self.ply_files = [self.ply_files[i] for i in indices[split_idx:]]

        print(f"Loaded {len(self.ply_files)} PLY files for {split} split")

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, index):
        # Load source point cloud
        src_pcd = o3d.io.read_point_cloud(self.ply_files[index])
        src_points = np.asarray(src_pcd.points)
        
        # Sample points if needed
        if len(src_points) > self.num_points:
            idx = np.random.choice(len(src_points), self.num_points, replace=False)
            src_points = src_points[idx]
        elif len(src_points) < self.num_points:
            idx = np.random.choice(len(src_points), self.num_points, replace=True)
            src_points = src_points[idx]

        # Generate random transformation
        rot = np.random.uniform(-self.rot_mag, self.rot_mag, 3)
        trans = np.random.uniform(-self.trans_mag, self.trans_mag, 3)
        
        # Create rotation matrix (using Euler angles)
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(np.radians(rot[0])), -np.sin(np.radians(rot[0]))],
                      [0, np.sin(np.radians(rot[0])), np.cos(np.radians(rot[0]))]])
        Ry = np.array([[np.cos(np.radians(rot[1])), 0, np.sin(np.radians(rot[1]))],
                      [0, 1, 0],
                      [-np.sin(np.radians(rot[1])), 0, np.cos(np.radians(rot[1]))]])
        Rz = np.array([[np.cos(np.radians(rot[2])), -np.sin(np.radians(rot[2])), 0],
                      [np.sin(np.radians(rot[2])), np.cos(np.radians(rot[2])), 0],
                      [0, 0, 1]])
        R = Rx @ Ry @ Rz

        # Apply transformation to get target point cloud
        tgt_points = (R @ src_points.T).T + trans

        # Apply noise based on noise_type
        if self.noise_type == 'jitter':
            # Add Gaussian noise
            src_points += np.random.normal(0, 0.02, src_points.shape)
            tgt_points += np.random.normal(0, 0.02, tgt_points.shape)
        elif self.noise_type == 'crop':
            # Randomly crop both point clouds
            src_keep = np.random.choice(len(src_points), 
                                      int(len(src_points) * self.partial_p_keep[0]), 
                                      replace=False)
            tgt_keep = np.random.choice(len(tgt_points), 
                                      int(len(tgt_points) * self.partial_p_keep[1]), 
                                      replace=False)
            src_points = src_points[src_keep]
            tgt_points = tgt_points[tgt_keep]

        # Convert to torch tensors
        src_points = torch.from_numpy(src_points.astype(np.float32))
        tgt_points = torch.from_numpy(tgt_points.astype(np.float32))
        R = torch.from_numpy(R.astype(np.float32))
        trans = torch.from_numpy(trans.astype(np.float32))

        return src_points[None], tgt_points[None], R[None], trans[None]
