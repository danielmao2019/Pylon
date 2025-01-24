from data.datasets import BaseDataset


class AirChangeDataset(BaseDataset):
    __doc__ = r"""
    Download:
    ```bash
        wget http://mplab.sztaki.hu/~bcsaba/test/SZTAKI_AirChange_Benchmark.zip
        unzip SZTAKI_AirChange_Benchmark.zip
        mv SZTAKI_AirChange_Benchmark AirChange
        rm SZTAKI_AirChange_Benchmark.zip
    ```
    """

    def _init_annotations(self) -> None:
        
