from data.datasets import BaseDataset


class SpaceNet2Dataset(BaseDataset):
    __doc__ = r"""
    Download:
        ```
        # setup AWS CLI
        conda create --name aws python=3.10 -y
        conda activate aws
        pip install awscli --upgrade --user
        export PATH=~/.local/bin:$PATH
        aws --version
        # download data
        mkdir <data-root>
        cd <data-root>
        aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_2_Vegas_test_public.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_3_Paris_test_public.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_4_Shanghai.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_4_Shanghai_test_public.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_5_Khartoum.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_5_Khartoum_test_public.tar.gz .
    """
