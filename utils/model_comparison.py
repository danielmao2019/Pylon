from typing import Dict, Any, Optional, Union
import torch


def compare_scores_vector(
    current_scores: Dict[str, Any],
    best_scores: Dict[str, Any],
    metric_directions: Dict[str, int]
) -> Optional[bool]:
    """
    Compare two score dictionaries using vector comparison (partial order).
    
    Uses first quadrant cone partial order: current >= best if and only if
    all entries of current are >= corresponding entries in best (after applying DIRECTION).
    
    Args:
        current_scores: Current epoch scores from scores['aggregated']
        best_scores: Best scores so far from scores['aggregated']  
        metric_directions: Dict mapping metric names to DIRECTION (+1 or -1)
        
    Returns:
        True if current_scores is strictly better than best_scores
        False if best_scores is strictly better than current_scores
        None if scores are incomparable
    """
    # Get common metric keys
    common_keys = set(current_scores.keys()) & set(best_scores.keys())
    if not common_keys:
        return None
    
    # Apply DIRECTION and compare each metric
    current_better = False
    best_better = False
    
    for key in common_keys:
        if key not in metric_directions:
            continue
            
        direction = metric_directions[key]
        current_val = current_scores[key]
        best_val = best_scores[key]
        
        # Handle tensor values
        if isinstance(current_val, torch.Tensor):
            current_val = current_val.item()
        if isinstance(best_val, torch.Tensor):
            best_val = best_val.item()
            
        # Apply direction (+1 = higher better, -1 = lower better)
        if direction == 1:
            if current_val > best_val:
                current_better = True
            elif current_val < best_val:
                best_better = True
        elif direction == -1:
            if current_val < best_val:
                current_better = True
            elif current_val > best_val:
                best_better = True
    
    # Check partial order results
    if current_better and not best_better:
        return True
    elif best_better and not current_better:
        return False
    else:
        return None  # Incomparable


def reduce_scores_to_scalar(
    scores: Dict[str, Any],
    order_config: Union[bool, Dict],
    metric_directions: Dict[str, int]
) -> float:
    """
    Reduce scores to scalar for best checkpoint selection.
    
    Args:
        scores: Score dictionary from scores['aggregated']
        order_config: 
            - True: equal-weight average of all scores
            - Dict: weighted combination with weights matching scores structure
        metric_directions: Dict mapping metric names to DIRECTION (+1 or -1)
        
    Returns:
        Single scalar value representing the scores
    """
    if order_config is True:
        # Equal-weight average of all scores
        weights = {key: 1.0 for key in scores.keys() if key in metric_directions}
    elif isinstance(order_config, dict):
        # Use provided weights
        weights = order_config.copy()
    else:
        raise ValueError(f"Invalid order_config: {order_config}")
    
    # Filter weights to only include valid metrics with directions
    valid_weights = {}
    for key, weight in weights.items():
        if key in scores and key in metric_directions:
            assert weight >= 0, f"Negative weight not allowed: {key}={weight}"
            valid_weights[key] = weight
    
    if not valid_weights:
        raise ValueError("No valid metrics found for score reduction")
    
    # Normalize weights to sum to 1
    total_weight = sum(valid_weights.values())
    if total_weight == 0:
        raise ValueError("All weights are zero")
    
    normalized_weights = {key: weight / total_weight for key, weight in valid_weights.items()}
    
    # Compute weighted average with DIRECTION applied
    weighted_sum = 0.0
    for key, weight in normalized_weights.items():
        direction = metric_directions[key]
        value = scores[key]
        
        # Handle tensor values
        if isinstance(value, torch.Tensor):
            value = value.item()
            
        # Apply direction and weight
        weighted_sum += direction * value * weight
    
    return weighted_sum


def compare_scores(
    current_scores: Dict[str, Any],
    best_scores: Dict[str, Any],
    order_config: Union[bool, Dict, None],
    metric_directions: Dict[str, int]
) -> bool:
    """
    Compare two score dictionaries based on order configuration.
    
    Args:
        current_scores: Current epoch scores from scores['aggregated']
        best_scores: Best scores so far from scores['aggregated']
        order_config:
            - False/None: vector comparison (partial order)
            - True: equal-weight average comparison
            - Dict: weighted average comparison
        metric_directions: Dict mapping metric names to DIRECTION (+1 or -1)
        
    Returns:
        True if current_scores is better than best_scores
    """
    if order_config is False or order_config is None:
        # Vector comparison
        result = compare_scores_vector(current_scores, best_scores, metric_directions)
        if result is None:
            # Incomparable - treat as "not improving"
            return False
        return result
    else:
        # Scalar comparison using reduction
        current_scalar = reduce_scores_to_scalar(current_scores, order_config, metric_directions)
        best_scalar = reduce_scores_to_scalar(best_scores, order_config, metric_directions)
        return current_scalar > best_scalar


def extract_metric_directions(metric) -> Dict[str, int]:
    """
    Extract DIRECTION attributes from metric classes.
    
    Args:
        metric: Metric instance or object with DIRECTION attribute
        
    Returns:
        Dict mapping metric names to DIRECTION values
    """
    directions = {}
    
    # Handle different metric types
    if hasattr(metric, 'DIRECTION'):
        # Single metric with DIRECTION
        directions['score'] = metric.DIRECTION
    elif hasattr(metric, '__dict__'):
        # Check for nested metrics or metric collections
        for attr_name, attr_value in metric.__dict__.items():
            if hasattr(attr_value, 'DIRECTION'):
                directions[attr_name] = attr_value.DIRECTION
    
    # Default fallback - assume all metrics should be maximized
    if not directions:
        directions = {'score': 1}
        
    return directions
