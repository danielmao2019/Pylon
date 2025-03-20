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
        self._weight_classes = dataset.weight_classes
        # No ponderation if weights for the corresponding number of class are available
        if len(self._weight_classes) != self._num_classes:
            self._weight_classes = None
        # self._weight_classes = None
        try:
            self._ignore_label = dataset.ignore_label
        except:
            self._ignore_label = None
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
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))

        self.last_feature = None
        self.visual_names = ["data_visual"]
        print('total : ' + str(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print('upconv : ' + str(sum(p.numel() for p in self.up_modules.parameters() if p.requires_grad)))
        print('downconv : ' + str(sum(p.numel() for p in self.down_modules.parameters() if p.requires_grad)))
        print(self._weight_classes)

    def set_class_weight(self,dataset):
        self._weight_classes = dataset.weight_classes
        # No ponderation if weights for the corresponding number of class are available
        if len(self._weight_classes) != self._num_classes:
            print('number of weights different of the number of classes')
            self._weight_classes = None

    def set_input(self, data):
        self.batch_idx = data.batch
        if isinstance(data, PairMultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
        else:
            self.pre_computed = None
            self.upsample = None
        if getattr(data, "pos_target", None) is not None:
            if isinstance(data, PairMultiScaleBatch):
                self.pre_computed_target = data.multiscale_target
                self.upsample_target = data.upsample_target
                del data.multiscale_target
                del data.upsample_target
            else:
                self.pre_computed_target = None
                self.upsample_target = None

            self.input0, self.input1 = data.to_data()
            self.batch_idx_target = data.batch_target
            self.labels = data.y
        else:
            self.input = data
            self.labels = None

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        data0 = self.input0
        data1 = self.input1

        #Layer 0 conv + EF
        data0 = self.down_modules[0](data0, precomputed=self.pre_computed)
        data1 = self.down_modules[0](data1, precomputed=self.pre_computed_target)
        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data1.x = data1.x - data0.x[nn_list[1, :], :]
        stack_down.append(data1)

        for i in range(1, len(self.down_modules) - 1):
            data1 = self.down_modules[i](data1, precomputed=self.pre_computed_target)
            stack_down.append(data1)

        #1024 : last layer
        data = self.down_modules[-1](data1, precomputed=self.pre_computed_target)

        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
        self.last_feature = data.x
        if self._use_category:
            self.output = self.FC_layer(self.last_feature, self.category)
        else:
            self.output = self.FC_layer(self.last_feature)

        self.data_visual = self.input1
        self.data_visual.pred = torch.max(self.output, -1)[1]

        return self.output

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
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))
            if cuda:
                self.FC_layer.cuda()
