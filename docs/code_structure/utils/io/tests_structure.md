# Utils IO Tests Structure

## Tests implementation structure

`tests/utils/io/chumpy/test_chumpy_loading.py`

```text
test_chumpy_loading.py
├── from utils.io.chumpy import load_chumpy
├── def test_a_chumpy_pickled_array_loads_as_the_array_it_carries
│   ├── # A file naming a chumpy class loads without chumpy installed, as the numpy array the pickled state carries.
│   ├── impls write a pickle whose reduce names a chumpy class and whose state holds a known array  # impls-node-one-step:skip
│   ├── calls load_chumpy
│   └── impls assert the loaded value is that array
├── def test_a_non_chumpy_value_survives_untouched
│   ├── # Only chumpy's own references are substituted, so every other value comes back as the stream produced it.
│   ├── impls write a pickle carrying an ordinary object beside the chumpy payload
│   ├── calls load_chumpy
│   └── impls assert that value is an instance of its own class rather than a stand-in
├── def test_the_load_leaves_sys_modules_untouched
│   ├── # The substitution is confined to the unpickler, so no chumpy entry is planted in the interpreter's module table.
│   ├── impls record the sys.modules keys naming chumpy
│   ├── calls load_chumpy
│   └── impls assert those keys are unchanged
└── def test_a_chumpy_payload_carrying_no_array_is_rejected
    ├── # A chumpy value whose state holds no array raises rather than resolving to something invented.
    ├── impls write a pickle whose chumpy payload holds no array
    └── impls assert load_chumpy raises TypeError
```
