from typing import Dict, Any, Optional, Union
import torch


def compare_scores_vector(
    current_scores: Dict[str, Any],
    best_scores: Dict[str, Any],
    metric_directions: Dict[str, Any]
) -> Optional[bool]:
    """
    Compare two score dictionaries using vector comparison (partial order).
    
    Uses first quadrant cone partial order: current >= best if and only if
    all entries of current are >= corresponding entries in best (after applying DIRECTION).
    
    Args:
        current_scores: Current epoch scores from scores['aggregated']
        best_scores: Best scores so far from scores['aggregated']  
        metric_directions: Dict mapping score keys to DIRECTION (+1 or -1), can be nested
        
    Returns:
        True if current_scores is strictly better than best_scores
        False if best_scores is strictly better than current_scores
        None if scores are incomparable
    """
    # Flatten nested directions for comparison
    flat_directions = _flatten_directions(metric_directions)
    
    # Flatten scores to match flattened directions
    flat_current = _flatten_scores(current_scores)
    flat_best = _flatten_scores(best_scores)
    
    # Get common metric keys
    common_keys = set(flat_current.keys()) & set(flat_best.keys())
    if not common_keys:
        return None
    
    # Apply DIRECTION and compare each metric
    current_better = False
    best_better = False
    
    for key in common_keys:
        # Look up direction for this key
        direction = flat_directions.get(key)
        assert direction is not None, f"No direction found for metric key '{key}' in {flat_directions}"
        current_val = flat_current[key]
        best_val = flat_best[key]
        
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
    metric_directions: Dict[str, Any]
) -> float:
    """
    Reduce scores to scalar for best checkpoint selection.
    
    Args:
        scores: Score dictionary from scores['aggregated']
        order_config: 
            - True: equal-weight average of all scores
            - Dict: weighted combination with weights matching scores structure
        metric_directions: Dict mapping score keys to DIRECTION (+1 or -1), can be nested
        
    Returns:
        Single scalar value representing the scores
    """
    # Flatten nested directions and scores for uniform processing
    flat_directions = _flatten_directions(metric_directions)
    flat_scores = _flatten_scores(scores)
    
    if order_config is True:
        # Equal-weight average of all scores
        weights = {key: 1.0 for key in flat_scores.keys()}
    elif isinstance(order_config, dict):
        # Use provided weights - flatten if needed
        flat_weights = _flatten_scores(order_config) if any(isinstance(v, dict) for v in order_config.values()) else order_config
        weights = flat_weights.copy()
    else:
        raise ValueError(f"Invalid order_config: {order_config}")
    
    # Filter weights to only include valid metrics 
    valid_weights = {}
    for key, weight in weights.items():
        if key in flat_scores:
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
        # Look up direction for this key
        direction = flat_directions.get(key)
        assert direction is not None, f"No direction found for metric key '{key}' in {flat_directions}"
        value = flat_scores[key]
        
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
    metric_directions: Dict[str, Any]
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
        metric_directions: Dict mapping metric names to DIRECTION (+1 or -1), can be nested
        
    Returns:
        True if current_scores is better than best_scores
    """
    # Unified comparison logic
    if order_config is False or order_config is None:
        # Vector comparison
        result = compare_scores_vector(current_scores, best_scores, metric_directions)
        if result is None:
            # Incomparable - treat as "not improving"
            return False
        return result
    
    # Scalar comparison using reduction (handles both True and Dict cases)
    current_scalar = reduce_scores_to_scalar(current_scores, order_config, metric_directions)
    best_scalar = reduce_scores_to_scalar(best_scores, order_config, metric_directions)
    return current_scalar > best_scalar


def get_metric_directions(metric) -> Dict[str, Any]:
    """
    Extract DIRECTIONS from metric classes.
    
    Args:
        metric: Metric instance with DIRECTIONS attribute
        
    Returns:
        Dict mapping score keys to DIRECTION values (+1 or -1)
        Can be nested dict structure matching the metric's output structure
    """
    assert metric is not None, "Metric cannot be None"
    
    # All metrics should now have explicit DIRECTIONS attribute
    if hasattr(metric, 'DIRECTIONS'):
        directions = metric.DIRECTIONS.copy()
        # Recursively validate all direction values
        _validate_directions(directions)
        return directions
    else:
        raise AttributeError(f"Metric {type(metric)} has no DIRECTIONS attribute. All metrics must define explicit DIRECTIONS = {{'score_key': 1 or -1}}")


def _validate_directions(directions: Union[Dict, int], path: str = "") -> None:
    """Recursively validate direction values are -1 or 1."""
    if isinstance(directions, dict):
        for key, value in directions.items():
            _validate_directions(value, f"{path}.{key}" if path else key)
    elif isinstance(directions, int):
        assert directions in [-1, 1], f"DIRECTION at '{path}' must be -1 or 1, got {directions}"
    else:
        raise TypeError(f"DIRECTION at '{path}' must be int or dict, got {type(directions)}")


def _flatten_directions(directions: Union[Dict, int], prefix: str = "") -> Dict[str, int]:
    """Flatten nested direction structure into flat dict with dotted keys."""
    if isinstance(directions, int):
        return {prefix: directions} if prefix else {"__default__": directions}
    
    flat = {}
    for key, value in directions.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_directions(value, new_prefix))
        else:
            flat[new_prefix] = value
    return flat


def _flatten_scores(scores: Union[Dict, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested score structure into flat dict with dotted keys."""
    if not isinstance(scores, dict):
        return {prefix: scores} if prefix else {"__default__": scores}
    
    flat = {}
    for key, value in scores.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_scores(value, new_prefix))
        else:
            flat[new_prefix] = value
    return flat


def extract_metric_directions(metric) -> Dict[str, int]:
    """
    Extract DIRECTION attributes from metric classes.
    
    Deprecated: Use get_metric_directions instead.
    """
    return get_metric_directions(metric)
