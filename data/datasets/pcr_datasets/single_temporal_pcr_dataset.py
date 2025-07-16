import os
import glob
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset


class SingleTemporalPCRDataset(SyntheticTransformPCRDataset):
    """Single-temporal point cloud registration dataset.
    
    This class handles single-temporal point cloud datasets where the same file
    is used for both source and target, typically for self-registration tasks.
    
    Features:
    - Single file used for both src and tgt (self-registration)
    - Transform-to-overlap cache system for synthetic pair generation
    - Parallel processing with deterministic seeding
    - No defensive programming - fail fast with clear errors
    """

    def _init_annotations(self) -> None:
        """Initialize file pair annotations for single-temporal dataset.
        
        For single-temporal, each file pair has same src_filepath and tgt_filepath.
        """
        # Find all point cloud files (utils/io/point_cloud.py handles all formats)
        all_files = []
        for pattern in ['*.ply', '*.las', '*.laz', '*.txt', '*.pth', '*.off']:
            all_files.extend(glob.glob(os.path.join(self.data_root, pattern)))
        pc_files = sorted(all_files)
        
        # Create file pair annotations - for single-temporal, src and tgt are the same file
        self.file_pair_annotations = []
        for file_path in pc_files:
            annotation = {
                'src_filepath': file_path,
                'tgt_filepath': file_path,  # Same file for self-registration
            }
            self.file_pair_annotations.append(annotation)
        
        print(f"Found {len(self.file_pair_annotations)} files for single-temporal PCR dataset")
