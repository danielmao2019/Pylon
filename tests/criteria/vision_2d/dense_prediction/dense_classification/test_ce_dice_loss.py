import pytest
import torch
from criteria.vision_2d.dense_prediction.dense_classification.ce_dice_loss import CEDiceLoss


@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 2, 3
    height, width = 32, 32

    # Create predicted logits with shape (N, C, H, W)
    y_pred = torch.randn(batch_size, num_classes, height, width, device=device)
    # Create ground truth with shape (N, H, W) with values in [0, num_classes)
    y_true = torch.randint(0, num_classes, (batch_size, height, width), device=device)

    return y_pred, y_true


def test_ce_dice_loss_basic(sample_data):
    """
    Test basic functionality of CEDiceLoss.
    """
    y_pred, y_true = sample_data
    device = y_pred.device

    # Initialize criterion
    criterion = CEDiceLoss().to(device)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_ce_dice_loss_perfect_predictions(sample_data):
    """
    Test CEDiceLoss with perfect predictions.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Create perfect predictions (one-hot encoded with high confidence)
    y_pred_perfect = torch.zeros_like(y_pred)
    for b in range(y_true.size(0)):
        y_pred_perfect[b].scatter_(0, y_true[b].unsqueeze(0), 100.0)  # High confidence for correct class

    # Initialize criterion
    criterion = CEDiceLoss().to(device)

    # Compute loss
    loss = criterion(y_pred_perfect, y_true)

    # Loss should be close to 0 for perfect predictions
    assert loss.item() < 0.1, f"Loss should be close to 0 for perfect predictions, got {loss.item()}"


def test_ce_dice_loss_with_class_weights(sample_data):
    """
    Test CEDiceLoss with class weights.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Create unequal class weights
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device)

    # Initialize criterion with weights
    criterion = CEDiceLoss(class_weights=class_weights).to(device)
    criterion_no_weights = CEDiceLoss().to(device)

    # Compute losses
    loss = criterion(y_pred, y_true)
    loss_no_weights = criterion_no_weights(y_pred, y_true)

    # Check that loss is still valid
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0
    
    # Loss should be different with unequal weights
    assert not torch.isclose(loss, loss_no_weights, rtol=1e-4).item()


def test_ce_dice_loss_with_ignore_index(sample_data):
    """
    Test CEDiceLoss with ignored pixels.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_index = 255

    # Create a version with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0, 0] = ignore_index

    # Initialize criterion with ignore_index
    criterion = CEDiceLoss(ignore_index=ignore_index).to(device)

    # Compute loss
    loss = criterion(y_pred, y_true_ignored)

    # Check that loss is still valid
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_ce_dice_loss_all_ignored(sample_data):
    """
    Test that an error is raised when all pixels are ignored.
    """
    y_pred, _ = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_index = 255

    # Create target with all pixels ignored
    y_true = torch.full_like(y_pred[:, 0], fill_value=ignore_index)

    # Initialize criterion
    criterion = CEDiceLoss(ignore_index=ignore_index).to(device)

    # Loss computation should raise an error when all pixels are ignored
    with pytest.raises(ValueError, match="No valid pixels found"):
        criterion(y_pred, y_true)


def test_ce_dice_loss_with_weights_and_ignore(sample_data):
    """
    Test CEDiceLoss with both class weights and ignore_index.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_index = 255

    # Create unequal class weights
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device)

    # Create a version with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0, 0] = ignore_index

    # Initialize criterion with weights and ignore_index
    criterion = CEDiceLoss(
        class_weights=class_weights,
        ignore_index=ignore_index
    ).to(device)

    # Compute loss
    loss = criterion(y_pred, y_true_ignored)

    # Check that loss is still valid
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_ce_dice_loss_alpha(sample_data):
    """
    Test CEDiceLoss with different alpha values.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Test different alpha values
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    losses = []

    for alpha in alphas:
        criterion = CEDiceLoss(alpha=alpha).to(device)
        loss = criterion(y_pred, y_true)
        losses.append(loss.item())

    # Different alpha values should result in different loss values
    # Alpha=0 is pure Dice, Alpha=1 is pure CE
    assert losses[0] != losses[-1], "Loss with alpha=0 should differ from alpha=1"
    
    # Check for monotonic trend as alpha increases (not guaranteed but common)
    increasing = all(losses[i] <= losses[i+1] for i in range(len(losses)-1))
    decreasing = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
    assert increasing or decreasing, "Losses should show a consistent trend with increasing alpha"


def test_ce_dice_loss_input_validation(sample_data):
    """
    Test input validation in CEDiceLoss.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Initialize criterion
    criterion = CEDiceLoss().to(device)

    # Test invalid shapes
    with pytest.raises(AssertionError):
        criterion(y_pred[:, :, :, :5], y_true)  # Width mismatch

    with pytest.raises(AssertionError):
        criterion(y_pred[:, :, :5], y_true)  # Height mismatch

    with pytest.raises(AssertionError):
        criterion(y_pred[0], y_true)  # Missing batch dimension

    # Test invalid values in y_true (greater than num_classes)
    y_true_invalid = y_true.clone()
    y_true_invalid[0, 0, 0] = num_classes + 10  # This should fail
    with pytest.raises(RuntimeError):
        criterion(y_pred, y_true_invalid)
