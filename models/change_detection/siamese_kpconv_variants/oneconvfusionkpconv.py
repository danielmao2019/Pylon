from typing import Any
from torch.nn import Sequential, Dropout, Linear
from torch import nn

from models.change_detection.siamese_kpconv_variants.common.base_modules import FastBatchNorm1d, MultiHeadClassifier
from models.change_detection.siamese_kpconv_variants.common.KPConv import *
from models.change_detection.siamese_kpconv_variants.common.partial_dense import *
from models.change_detection.siamese_kpconv_variants.common.unet import UnwrappedUnetBasedModel
from models.change_detection.siamese_kpconv_variants.common.pair import PairMultiScaleBatch
from models.change_detection.siamese_kpconv_variants.common.torch_cluster.knn import knn


class OneConvFusionKPConv(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
        self._use_category = getattr(option, "use_category", False)
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._class_to_seg = dataset.class_to_segments
            self._num_categories = len(self._class_to_seg)
        else:
            self._num_categories = 0

        # Assemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Build final MLP
        self.last_mlp_opt = option.mlp_cls
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                self.last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=self.last_mlp_opt.dropout,
                bn_momentum=self.last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = self.last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(self.last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, self.last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(self.last_mlp_opt.nn[i], momentum=self.last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = self.last_mlp_opt.nn[i]

            if self.last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=self.last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))

        self.last_feature = None
        print('total : ' + str(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print('upconv : ' + str(sum(p.numel() for p in self.up_modules.parameters() if p.requires_grad)))
        print('downconv : ' + str(sum(p.numel() for p in self.down_modules.parameters() if p.requires_grad)))

    def forward(self, data) -> Any:
        # Process input data
        batch_idx = data.batch
        if isinstance(data, PairMultiScaleBatch):
            pre_computed = data.multiscale
            upsample = data.upsample
        else:
            pre_computed = None
            upsample = None

        if getattr(data, "pos_target", None) is not None:
            if isinstance(data, PairMultiScaleBatch):
                pre_computed_target = data.multiscale_target
                upsample_target = data.upsample_target
                del data.multiscale_target
                del data.upsample_target
            else:
                pre_computed_target = None
                upsample_target = None

            input0, input1 = data.to_data()
            batch_idx_target = data.batch_target
            labels = data.y
        else:
            input0 = data
            input1 = None
            batch_idx_target = None
            labels = None
            pre_computed_target = None
            upsample_target = None

        stack_down = []

        #Layer 0 conv + EF
        data0 = self.down_modules[0](input0, precomputed=pre_computed)
        data1 = self.down_modules[0](input1, precomputed=pre_computed_target)
        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data1.x = data1.x - data0.x[nn_list[1, :], :]
        stack_down.append(data1)

        for i in range(1, len(self.down_modules) - 1):
            data1 = self.down_modules[i](data1, precomputed=pre_computed_target)
            stack_down.append(data1)

        #1024 : last layer
        data = self.down_modules[-1](data1, precomputed=pre_computed_target)

        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=upsample_target)
        
        self.last_feature = data.x
        if self._use_category:
            output = self.FC_layer(self.last_feature, self.category)
        else:
            output = self.FC_layer(self.last_feature)

        return output

    def reset_final_layer(self, cuda = True):
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                self.last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=self.last_mlp_opt.dropout,
                bn_momentum=self.last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = self.last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(self.last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, self.last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(self.last_mlp_opt.nn[i], momentum=self.last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = self.last_mlp_opt.nn[i]

            if self.last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=self.last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            if cuda:
                self.FC_layer.cuda()
