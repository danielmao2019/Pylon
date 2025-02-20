# import pytest
# from .basenet_lwganet_l2 import BaseNet_LWGANet_L2
# import torch

# def test_lwganet_l2() -> None:
#     model = BaseNet_LWGANet_L2(preptrained_path='/pub7/yuchen/Pylon/models/change_detection/lwganet/lwganet_l2_e296.pth').to('cuda')
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     inputs = {
#         'img_1': torch.zeros(size=(16, 3, 32, 32)).to('cuda'),
#         'img_2': torch.zeros(size=(16, 3, 32, 32)).to('cuda'),
#     }
#     output = model(inputs)
#     assert output.shape == torch.Size([16, 2, 32, 32]), f'{output.shape=}'
    
import pytest
from .basenet_lwganet_l2 import BaseNet_LWGANet_L2
import torch
import torch.distributed as dist

def test_lwganet_l2() -> None:
    # Initialize the default process group if not already initialized.
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:23456',
            rank=0,
            world_size=1
        )
        
        #add to model init
    
    model = BaseNet_LWGANet_L2(
        preptrained_path='/pub7/yuchen/Pylon/models/change_detection/lwganet/lwganet_l2_e296.pth'
    ).to('cuda')
    
    # Convert model's BatchNorm layers to SyncBatchNorm.
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Increase the input size to avoid excessive downsampling
    inputs = {
        'img_1': torch.zeros(size=(16, 3, 128, 128)).to('cuda'),
        'img_2': torch.zeros(size=(16, 3, 128, 128)).to('cuda'),
    }
    
    result = model(inputs)
    #check paper to determine model output
    
    
    assert result['mask_p2'].shape == torch.Size([16, 1, 128, 128])
    assert result['mask_p3'].shape == torch.Size([16, 1, 128, 128])
    assert result['mask_p4'].shape == torch.Size([16, 1, 128, 128])
    assert result['mask_p5'].shape == torch.Size([16, 1, 128, 128])
    
    # Clean up the process group.
    dist.destroy_process_group()
    
    '''
    for reference 
    
    # run the model
        output, output2, output3, output4 = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var) + BCEDiceLoss(output2, target_var) + \
               BCEDiceLoss(output3, target_var) + BCEDiceLoss(output4, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()
        
    def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice
        
    '''
