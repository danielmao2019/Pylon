from typing import Optional
import copy
import torch
from models.change_detection.ftn.modules.swin.swin_trans_encoder import SwinTransEncoder


class encoder1(torch.nn.Module):
    def __init__(self, pretrained_path: Optional[str] = "./models/change_detection/ftn/swin_pretrain_224.pth"):
        super(encoder1, self).__init__()
        self.encoder1 = SwinTransEncoder(img_size=224, patch_size=4, in_chans=3, num_classes=2, embed_dim=128,
                                         depths=[2, 2, 18, 2], depths_decoder=[4, 4, 4, 4], num_heads=[4, 8, 16, 32],
                                         window_size=7)
        self.pretrained_path = pretrained_path
        self.load_from()

    def load_from(self):
        pretrained_path = self.pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.encoder1.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.encoder1.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    # print(1)
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.encoder1.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")

    def forward(self, img1, img2):
        x, y1, y2 = self.encoder1(img1, img2)
        return x, y1, y2
