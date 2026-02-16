import json
from typing import Any, Dict, List, Tuple

from dash import callback_context
from dash._utils import stringify_id


def triggered_payload() -> Tuple[Any, Any]:
    """Return (triggered_id, triggered_value) from the active Dash context."""
    ctx = callback_context
    triggered = (
        ctx.triggered[0] if ctx.triggered else {'prop_id': 'initial', 'value': None}
    )
    triggered_prop = triggered['prop_id']
    triggered_id_raw = triggered_prop.split('.')[0]
    triggered_id = _decode_trigger_id(triggered_id_raw)
    assert 'value' in triggered, "Dash triggered payload missing value"
    triggered_value = triggered['value']
    return triggered_id, triggered_value


def parse_model_state(
    store_state_entries: List[Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    # Input validations
    assert isinstance(
        store_state_entries, list
    ), f"store_state_entries must be list, got {type(store_state_entries)}"

    model_state: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for state_entry in store_state_entries:
        if isinstance(state_entry, list):
            state_items = state_entry
        elif isinstance(state_entry, dict):
            state_items = [state_entry]
        else:
            raise AssertionError(
                f"Unexpected store state entry type {type(state_entry)}; expected list or dict"
            )
        for state_item in state_items:
            # --- Validation ---
            assert isinstance(state_item, dict), "State entry must be a dict"
            assert (
                'id' in state_item and 'property' in state_item
            ), f"State entry must include 'id' and 'property', got keys {set(state_item.keys())}"
            assert not (
                set(state_item.keys()) - {'id', 'property', 'value'}
            ), f"Unexpected state entry keys: {set(state_item.keys())}"
            assert isinstance(state_item['id'], dict), "State id must be a dict"
            assert set(state_item['id'].keys()) == {
                'type',
                'dataset',
                'scene',
                'method',
                'field',
            }, f"Unexpected state id keys: {set(state_item['id'].keys())}"
            assert (
                state_item['id']['type'] == 'model-store'
            ), "State id type must be 'model-store'"
            assert (
                state_item['property'] == 'data'
            ), f"Store property must be 'data', got {state_item['property']}"
            assert (
                'value' in state_item
                or callback_context.states[
                    f"{stringify_id(state_item['id'])}.{state_item['property']}"
                ]
                is None
            ), "State entry missing 'value' but context provides non-None data"

            # --- Parsing and aggregation ---
            state_id = state_item['id']
            property_name = state_item['property']
            state_key = f"{stringify_id(state_id)}.{property_name}"
            context_value = callback_context.states[state_key]
            if 'value' in state_item:
                store_value = state_item['value']
                assert (
                    context_value == store_value
                ), f"Context state {state_key} does not match entry value"
            else:
                assert (
                    context_value is None
                ), f"State entry missing 'value' but context has {context_value}"
                store_value = context_value
            dataset_key = state_id['dataset']
            scene_key = state_id['scene']
            method_key = state_id['method']
            field_key = state_id['field']
            model_state.setdefault(dataset_key, {}).setdefault(
                scene_key, {}
            ).setdefault(method_key, {})[field_key] = store_value

    return model_state


def _decode_trigger_id(raw: str) -> Any:
    if raw.startswith('{'):
        return json.loads(raw)
    return raw
