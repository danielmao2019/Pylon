from typing import Dict, Any
import torch
from torch.nn import Sequential, Dropout, Linear
from torch import nn

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_geometric.nn import knn


class SiameseKPConv(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
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
            print("Using category information for the predictions with %i categories", self._num_categories)
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
                            nn.LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = self.last_mlp_opt.nn[i]

            if self.last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=self.last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))

        self.visual_names = ["data_visual"]
        print('total : ' + str(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print('upconv : ' + str(sum(p.numel() for p in self.up_modules.parameters() if p.requires_grad)))
        print('downconv : ' + str(sum(p.numel() for p in self.down_modules.parameters() if p.requires_grad)))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        data0 = inputs['pc_0']
        data1 = inputs['pc_1']

        for i in range(len(self.down_modules) - 1):
            data0 = self.down_modules[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules[i](data1, precomputed=self.pre_computed_target)
            nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
            diff = data1.clone()
            diff.x = data1.x - data0.x[nn_list[1,:],:]
            stack_down.append(diff)
        #1024 : last layer
        data0 = self.down_modules[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules[-1](data1, precomputed=self.pre_computed_target)
        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data = data1.clone()
        data.x = data1.x - data0.x[nn_list[1,:],:]
        innermost = False
        if not isinstance(self.inner_modules[0], torch.nn.Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
        last_feature = data.x
        if self._use_category:
            output = self.FC_layer(last_feature, self.category)
        else:
            output = self.FC_layer(last_feature)

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
                            nn.LeakyReLU(0.2),
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
