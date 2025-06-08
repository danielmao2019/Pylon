from data.datasets.base_dataset import BaseDataset


class COCOStuff164K(BaseDataset):
    __doc__ = r"""Reference:

    Download:
        Reference:
            https://github.com/xu-ji/IIC/blob/master/datasets/setup_cocostuff164k.sh
            https://github.com/xu-ji/IIC/blob/master/datasets/README.txt
        Steps:
            cd <data-root>
            wget -nc -P . http://images.cocodataset.org/zips/train2017.zip
            wget -nc -P . http://images.cocodataset.org/zips/val2017.zip
            wget -nc -P . http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
            mkdir -p ./images
            mkdir -p ./annotations
            unzip -n ./train2017.zip -d ./images/
            unzip -n ./val2017.zip -d ./images/
            unzip -n ./stuffthingmaps_trainval2017.zip -d ./annotations/
            wget https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz
            tar -xzvf COCOStuff164kCurated.tar.gz
            mv COCO/CocoStuff164k/curated .
            rmdir COCO/CocoStuff164k
            rmdir COCO
    """
