"""Shared fixtures and helper functions for OSCD dataset tests."""

import pytest
import os


@pytest.fixture
def create_dummy_oscd_files():
    """Fixture that returns a function to create minimal OSCD dataset structure for testing."""
    def _create_files(temp_dir: str) -> None:
        """Create minimal OSCD dataset structure for testing."""
        # Create required directory structure
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Create split files
        train_labels_dir = os.path.join(temp_dir, 'train_labels')
        test_labels_dir = os.path.join(temp_dir, 'test_labels')
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(test_labels_dir, exist_ok=True)
        
        # Create train.txt and test.txt files
        with open(os.path.join(images_dir, 'train.txt'), 'w') as f:
            f.write('aguasclaras,cupertino\n')
        
        with open(os.path.join(images_dir, 'test.txt'), 'w') as f:
            f.write('bastia,montpellier\n')
            
        # Create city directories for train cities
        for city in ['aguasclaras', 'cupertino']:
            city_dir = os.path.join(images_dir, city)
            
            # Create imgs_1 and imgs_2 directories with dummy TIF files
            for img_phase in ['imgs_1', 'imgs_2']:
                phase_dir = os.path.join(city_dir, img_phase)
                os.makedirs(phase_dir, exist_ok=True)
                
                # Create dummy TIF files efficiently
                import subprocess
                tif_files = [os.path.join(phase_dir, f'patch_{i+1}.tif') for i in range(2)]
                subprocess.run(['touch'] + tif_files, check=True)
            
            # Create pair directory with PNG files
            pair_dir = os.path.join(city_dir, 'pair')
            os.makedirs(pair_dir, exist_ok=True)
            
            # Create PNG files efficiently
            png_files = [os.path.join(pair_dir, img_file) for img_file in ['img1.png', 'img2.png']]
            subprocess.run(['touch'] + png_files, check=True)
            
            # Create dates.txt file
            dates_file = os.path.join(city_dir, 'dates.txt')
            with open(dates_file, 'w') as f:
                f.write('date_1: 20180101\n')
                f.write('date_2: 20181201\n')
            
            # Create labels with cm subdirectory
            train_city_labels = os.path.join(train_labels_dir, city)
            cm_dir = os.path.join(train_city_labels, 'cm')
            os.makedirs(cm_dir, exist_ok=True)
            
            # Create label files efficiently
            label_files = [
                os.path.join(cm_dir, f'{city}-cm.tif'),
                os.path.join(cm_dir, 'cm.png')
            ]
            subprocess.run(['touch'] + label_files, check=True)
        
        # Create city directories for test cities
        for city in ['bastia', 'montpellier']:
            city_dir = os.path.join(images_dir, city)
            
            # Create imgs_1 and imgs_2 directories with dummy TIF files
            for img_phase in ['imgs_1', 'imgs_2']:
                phase_dir = os.path.join(city_dir, img_phase)
                os.makedirs(phase_dir, exist_ok=True)
                
                # Create dummy TIF files efficiently
                tif_files = [os.path.join(phase_dir, f'patch_{i+1}.tif') for i in range(2)]
                subprocess.run(['touch'] + tif_files, check=True)
            
            # Create pair directory with PNG files
            pair_dir = os.path.join(city_dir, 'pair')
            os.makedirs(pair_dir, exist_ok=True)
            
            # Create PNG files efficiently
            png_files = [os.path.join(pair_dir, img_file) for img_file in ['img1.png', 'img2.png']]
            subprocess.run(['touch'] + png_files, check=True)
            
            # Create dates.txt file
            dates_file = os.path.join(city_dir, 'dates.txt')
            with open(dates_file, 'w') as f:
                f.write('date_1: 20180101\n')
                f.write('date_2: 20181201\n')
            
            # Create labels with cm subdirectory
            test_city_labels = os.path.join(test_labels_dir, city)
            cm_dir = os.path.join(test_city_labels, 'cm')
            os.makedirs(cm_dir, exist_ok=True)
            
            # Create label files efficiently
            label_files = [
                os.path.join(cm_dir, f'{city}-cm.tif'),
                os.path.join(cm_dir, 'cm.png')
            ]
            subprocess.run(['touch'] + label_files, check=True)
    
    return _create_files
