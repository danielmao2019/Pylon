import torch
import torchvision


def get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
) -> torch.nn.ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        pretrained=pretrained
    ).features

    # Slicing it:
    derived_model = torch.nn.ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model
