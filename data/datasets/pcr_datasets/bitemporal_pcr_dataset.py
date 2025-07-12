import os
import glob
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset


class BitemporalPCRDataset(SyntheticTransformPCRDataset):
    """Bi-temporal point cloud registration dataset for multi-scene datasets.
    
    This class handles real point cloud pairs from different files/times,
    using the transform-to-overlap cache system for synthetic pair generation.
    
    Features:
    - Multi-scene point cloud pairs (different files for src/tgt)
    - Transform-to-overlap cache system for synthetic pair generation
    - Parallel processing with deterministic seeding
    - No defensive programming - fail fast with clear errors
    """
    
    def _init_annotations(self) -> None:
        """Initialize file pair annotations for bi-temporal dataset.
        
        For bi-temporal, each file pair has different src_file_path and tgt_file_path.
        This is a base implementation - subclasses should override for specific datasets.
        """
        # Find all point cloud files (utils/io/point_cloud.py handles all formats)
        all_files = []
        for pattern in ['*.ply', '*.las', '*.laz', '*.txt', '*.pth', '*.off']:
            all_files.extend(glob.glob(os.path.join(self.data_root, pattern)))
        pc_files = sorted(all_files)
        
        # Create simple adjacent pairs for demonstration
        # Real datasets would have more sophisticated pairing logic (metadata, timestamps, etc.)
        self.file_pair_annotations = []
        
        for i in range(len(pc_files) - 1):
            annotation = {
                'src_file_path': pc_files[i],
                'tgt_file_path': pc_files[i + 1],  # Different file for bi-temporal
            }
            self.file_pair_annotations.append(annotation)
        
        print(f"Found {len(self.file_pair_annotations)} file pairs for bi-temporal PCR dataset")

