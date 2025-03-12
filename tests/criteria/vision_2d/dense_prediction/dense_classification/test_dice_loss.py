import pytest
import torch
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss


@pytest.fixture
def y_pred():
    return torch.tensor([
        [[[0.0386, 0.9645, 0.8292, 0.0196, 0.3464],
          [0.7008, 0.8797, 0.0763, 0.2584, 0.3067],
          [0.7858, 0.8736, 0.5556, 0.5184, 0.8900],
          [0.8904, 0.1982, 0.4975, 0.1508, 0.6437],
          [0.9239, 0.5227, 0.8201, 0.0126, 0.4945]],

         [[0.0402, 0.7018, 0.2901, 0.6005, 0.1142],
          [0.3373, 0.3594, 0.0833, 0.6761, 0.9218],
          [0.6617, 0.7649, 0.3870, 0.6312, 0.1959],
          [0.7572, 0.1285, 0.9468, 0.3012, 0.7814],
          [0.4879, 0.6178, 0.1224, 0.6696, 0.4125]]],


        [[[0.9350, 0.1163, 0.8317, 0.9510, 0.9385],
          [0.6242, 0.5782, 0.6960, 0.6504, 0.7436],
          [0.8114, 0.4750, 0.1006, 0.4506, 0.0960],
          [0.0144, 0.9185, 0.3062, 0.7547, 0.1604],
          [0.6598, 0.9908, 0.5493, 0.3642, 0.9267]],

         [[0.0735, 0.2730, 0.7054, 0.8376, 0.0693],
          [0.0666, 0.5842, 0.7272, 0.4409, 0.2009],
          [0.0025, 0.8686, 0.7557, 0.5897, 0.4675],
          [0.3866, 0.1547, 0.3830, 0.5304, 0.0890],
          [0.9657, 0.9450, 0.3341, 0.3168, 0.5159]]],


        [[[0.8872, 0.8578, 0.9395, 0.9941, 0.2993],
          [0.8751, 0.6884, 0.3923, 0.0922, 0.5593],
          [0.6478, 0.1098, 0.2289, 0.0626, 0.5936],
          [0.0241, 0.6682, 0.0736, 0.0340, 0.0177],
          [0.8507, 0.1964, 0.3351, 0.2613, 0.7168]],

         [[0.8919, 0.0588, 0.8948, 0.6937, 0.7473],
          [0.1518, 0.0646, 0.3452, 0.3778, 0.9921],
          [0.0503, 0.0808, 0.0852, 0.1802, 0.1175],
          [0.0379, 0.4512, 0.8742, 0.7042, 0.4112],
          [0.2072, 0.1684, 0.1351, 0.1274, 0.7180]]]])


@pytest.fixture
def y_true():
    return torch.tensor([
        [[0, 0, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 0, 1]],

        [[0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0],
         [1, 0, 1, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 1, 1, 0]],

        [[1, 1, 0, 1, 0],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 1, 1],
         [1, 1, 1, 0, 1],
         [1, 1, 0, 1, 1]]])


@pytest.fixture
def expected():
    return torch.tensor(0.5063)


def test_dice_loss_basic(y_pred, y_true, expected):
    criterion = DiceLoss()
    assert torch.isclose(criterion(y_pred, y_true), expected, rtol=1e-4).item()


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_dice_loss_reduction(y_pred, y_true, reduction):
    criterion = DiceLoss(reduction=reduction)
    loss = criterion(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() > 0  # Dice loss is always positive


def test_dice_loss_with_class_weights(y_pred, y_true):
    # Test with equal class weights (should be same as no weights)
    criterion_equal = DiceLoss(class_weights=torch.ones(2))
    loss_equal = criterion_equal(y_pred, y_true)
    
    criterion_none = DiceLoss()
    loss_none = criterion_none(y_pred, y_true)
    
    assert torch.isclose(loss_equal, loss_none, rtol=1e-4).item()
    
    # Test with unequal class weights
    criterion_unequal = DiceLoss(class_weights=torch.tensor([0.2, 0.8]))
    loss_unequal = criterion_unequal(y_pred, y_true)
    
    # Loss should be different with unequal weights
    assert not torch.isclose(loss_unequal, loss_none, rtol=1e-4).item()


def test_dice_loss_with_ignore_index(y_pred, y_true):
    # Create a version of y_true with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0, 0] = 255  # Set one pixel to ignore_index
    
    criterion = DiceLoss(ignore_index=255)
    loss = criterion(y_pred, y_true_ignored)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_dice_loss_perfect_predictions():
    batch_size, num_classes = 2, 3
    height, width = 32, 32
    criterion = DiceLoss()

    # Create ground truth labels
    y_true = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Create perfect predictions (one-hot encoded with high confidence)
    y_pred = torch.zeros(batch_size, num_classes, height, width)
    for b in range(batch_size):
        y_pred[b].scatter_(0, y_true[b].unsqueeze(0), 100.0)  # High confidence for correct class

    # Compute loss
    loss = criterion(y_pred, y_true)

    # For perfect predictions with high confidence, loss should be very small
    assert loss.item() < 0.01


def test_dice_loss_all_ignored():
    criterion = DiceLoss(ignore_index=255)
    batch_size, num_classes = 2, 3
    height, width = 32, 32

    # Create predictions and targets where all pixels are ignored
    y_pred = torch.randn(batch_size, num_classes, height, width)
    y_true = torch.full((batch_size, height, width), fill_value=255)

    # Loss computation should raise an error when all pixels are ignored
    with pytest.raises(ValueError, match="All pixels in target are ignored"):
        criterion(y_pred, y_true)
