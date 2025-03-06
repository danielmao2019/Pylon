from collections import OrderedDict
from typing import Dict, List
import os
import torch

from torch_points3d.core.regularizer import *
from torch_points3d.utils.colors import COLORS


class BaseModel(torch.nn.Module):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    __REQUIRED_DATA__: List[str] = []
    __REQUIRED_LABELS__: List[str] = []

    def __init__(self, opt):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        super(BaseModel, self).__init__()
        self.opt = opt
        self.visual_names = []
        self.output = None
        self._conv_type = opt.conv_type if hasattr(opt, "conv_type") else None  # Update to OmegaConv 2.0
        self._spatial_ops_dict: Dict = {}

    @property
    def conv_type(self):
        return self._conv_type

    @conv_type.setter
    def conv_type(self, conv_type):
        self._conv_type = conv_type

    def load_state_dict_with_same_shape(self, weights, strict=False):
        model_state = self.state_dict()
        filtered_weights = {k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()}
        print("Loading weights:" + ", ".join(filtered_weights.keys()))
        self.load_state_dict(filtered_weights, strict=strict)

    def set_pretrained_weights(self):
        path_pretrained = getattr(self.opt, "path_pretrained", None)
        weight_name = getattr(self.opt, "weight_name", "latest")

        if path_pretrained is not None:
            if not os.path.exists(path_pretrained):
                print("The path does not exist, it will not load any model")
            else:
                print("load pretrained weights from {}".format(path_pretrained))
                m = torch.load(path_pretrained, map_location="cpu")["models"][weight_name]
                self.load_state_dict_with_same_shape(m, strict=False)

    def _manage_optimizer_zero_grad(self):
        if not self._accumulated_gradient_step:
            self._optimizer.zero_grad()  # clear existing gradients
            return True
        else:
            if self._accumulated_gradient_count == self._accumulated_gradient_step:
                self._accumulated_gradient_count = 0
                return True

            if self._accumulated_gradient_count == 0:
                self._optimizer.zero_grad()  # clear existing gradients
            self._accumulated_gradient_count += 1
            return False

    def get_spatial_ops(self):
        return self._spatial_ops_dict

    def enable_dropout_in_eval(self):
        def search_from_key(modules):
            for _, m in modules.items():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
                search_from_key(m._modules)

        search_from_key(self._modules)

    def get_from_opt(self, opt, keys=[], default_value=None, msg_err=None, silent=True):
        if len(keys) == 0:
            raise Exception("Keys should not be empty")
        value_out = default_value

        def search_with_keys(args, keys, value_out):
            if len(keys) == 0:
                value_out = args
                return value_out
            value = args[keys[0]]
            return search_with_keys(value, keys[1:], value_out)

        try:
            value_out = search_with_keys(opt, keys, value_out)
        except Exception as e:
            if msg_err:
                raise Exception(str(msg_err))
            else:
                if not silent:
                    print(e)
            value_out = default_value
        return value_out

    def get_current_visuals(self):
        """Return an OrderedDict containing associated tensors within visual_names"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def verify_data(self, data, forward_only=False):
        """Goes through the __REQUIRED_DATA__ and __REQUIRED_LABELS__ attribute of the model
        and verifies that the passed data object contains all required members.
        If something is missing it raises a KeyError exception.
        """
        missing_keys = []
        required_attributes = self.__REQUIRED_DATA__
        if not forward_only:
            required_attributes += self.__REQUIRED_LABELS__
        for attr in required_attributes:
            if not hasattr(data, attr) or data[attr] is None:
                missing_keys.append(attr)
        if len(missing_keys):
            raise KeyError(
                "Missing attributes in your data object: {}. The model will fail to forward.".format(missing_keys)
            )

    def print_transforms(self):
        message = ""
        for attr in self.__dict__:
            if "transform" in attr:
                message += "{}{} {}= {}\n".format(COLORS.IPurple, attr, COLORS.END_NO_TOKEN, getattr(self, attr))
        print(message)
