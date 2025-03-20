from typing import Any
from torch.nn import Sequential, Dropout, Linear
from torch import nn
import copy

from models.change_detection.siamese_kpconv_variants.common.base_modules import FastBatchNorm1d, MultiHeadClassifier
from models.change_detection.siamese_kpconv_variants.common.KPConv import *
from models.change_detection.siamese_kpconv_variants.common.partial_dense import *
from models.change_detection.siamese_kpconv_variants.common.unet import UnwrappedUnetBasedModel
from models.change_detection.siamese_kpconv_variants.common.pair import PairMultiScaleBatch
from models.change_detection.siamese_kpconv_variants.common.torch_cluster.knn import knn


class BaseFactoryPSI:
    def __init__(self, module_name_down_1, module_name_down_2, module_name_up, modules_lib):
        self.module_name_down_1 = module_name_down_1
        self.module_name_down_2 = module_name_down_2
        self.module_name_up = module_name_up
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP":
            return getattr(self.modules_lib, self.module_name_up, None)
        elif "1" in flow:
            return getattr(self.modules_lib, self.module_name_down_1, None)
        else:
            return getattr(self.modules_lib, self.module_name_down_2, None)


####################SIAMESE KP CONV UNSHARED (PSEUDO SIAMESE)############################
class SiameseKPConvUnshared(UnwrappedUnetBasedModel):
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
        opt = copy.deepcopy(option)
        super(UnwrappedUnetBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [], "upsample_op": []}

        self._init_from_compact_format(opt, model_type, dataset, modules)

        # Unshared weight :  2 down modules
        # Build final MLP
        last_mlp_opt = option.mlp_cls
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=last_mlp_opt.dropout,
                bn_momentum=last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            nn.LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = last_mlp_opt.nn[i]

            if last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))

        self.visual_names = ["data_visual"]

    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """
        self.down_modules_1 = nn.ModuleList()
        self.down_modules_2 = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_modules = nn.ModuleList()

        self.save_sampling_id_1 = opt.down_conv_1.get('save_sampling_id')
        self.save_sampling_id_2 = opt.down_conv_2.get('save_sampling_id')

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name_1 = opt.down_conv_1.module_name
        down_conv_cls_name_2 = opt.down_conv_2.module_name
        up_conv_cls_name = opt.up_conv.module_name if opt.get('up_conv') is not None else None
        self._factory_module = factory_module_cls(
            down_conv_cls_name_1, down_conv_cls_name_2, up_conv_cls_name, modules_lib
        )  # Create the factory object

        # Loal module
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            inners = self._create_inner_modules(opt.innermost, modules_lib)
            for inner in inners:
                self.inner_modules.append(inner)
        else:
            self.inner_modules.append(Identity())

        # Down modules
        for i in range(len(opt.down_conv_1.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_1, i, "DOWN_1")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_1.append(down_module)
        for i in range(len(opt.down_conv_2.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_2, i, "DOWN_2")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_2.append(down_module)

        # Up modules
        if up_conv_cls_name:
            for i in range(len(opt.up_conv.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv, i, "UP")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules.append(up_module)

    def _get_factory(self, model_name, modules_lib) -> BaseFactoryPSI:
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            factory_module_cls = BaseFactoryPSI
        return factory_module_cls

    def forward(self, data, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
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

        data0 = input0
        data1 = input1

        for i in range(len(self.down_modules_1) - 1):
            data0 = self.down_modules_1[i](data0, precomputed=pre_computed)
            data1 = self.down_modules_2[i](data1, precomputed=pre_computed_target)
            nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
            diff = data1.clone()
            diff.x = data1.x - data0.x[nn_list[1,:],:]
            stack_down.append(diff)
        #1024
        data0 = self.down_modules_1[-1](data0, precomputed=pre_computed)
        data1 = self.down_modules_2[-1](data1, precomputed=pre_computed_target)

        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data = data1.clone()
        data.x = data1.x - data0.x[nn_list[1,:],:]
        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data1)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=upsample_target)
        last_feature = data.x
        if self._use_category:
            output = self.FC_layer(last_feature, self.category)
        else:
            output = self.FC_layer(last_feature)

        self.data_visual = input1
        self.data_visual.pred = torch.max(output, -1)[1]

        return output
