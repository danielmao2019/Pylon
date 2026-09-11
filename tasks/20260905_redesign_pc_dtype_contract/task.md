goal: re-design pc dtype contract/provenance

## Table of Contents <!-- omit in toc -->

- [1. Guidelines](#1-guidelines)
  - [1.1. Proposed Solution](#11-proposed-solution)
    - [1.1.1. Type Casting](#111-type-casting)
    - [1.1.2. Color Data Convention Conversion](#112-color-data-convention-conversion)
    - [1.1.3. Layout Mapping](#113-layout-mapping)
    - [1.1.4. New Meta Data API](#114-new-meta-data-api)
    - [1.1.5. Point Cloud Data Structure Construction and I/O](#115-point-cloud-data-structure-construction-and-io)
    - [1.1.6. What Becomes Stale Design](#116-what-becomes-stale-design)
    - [1.1.7. Seriously Bad Behavior Observed when Working on this Task](#117-seriously-bad-behavior-observed-when-working-on-this-task)
  - [1.2. Solution Constraints](#12-solution-constraints)
- [2. Definition of Done](#2-definition-of-done)
  - [2.1. Project Consumers be Refactored](#21-project-consumers-be-refactored)
  - [2.2. Task Scope](#22-task-scope)

----------

## 1. Guidelines

### 1.1. Proposed Solution

#### 1.1.1. Type Casting

1. the fundamental root cause is the dtype system mismatch: the dtype systems are each a subset of one universal, system-agnostic collection of conceptual dtypes, and no system's subset contains every other's.
   1. conceptual dtype identity across systems:
      1. uint16 and int32 are two distinct conceptual dtypes. numpy int32 and torch int32 represent the same conceptual dtype.
      2. every ply dtype torch carries loads unchanged: i1 as int8, u1 as uint8, i2 as int16, i4 as int32, f4 as float32, f8 as float64.
   2. each system's supported subset:
      1. ply's subset is i1, u1, i2, u2, i4, u4, f4 and f8, so ply has no 64-bit integer and no boolean.
      2. torch 2.2.2 has no uint16, uint32, uint64 or float128, and bfloat16 is its alone.
      3. numpy 1.26.4 has uint64 and float128, and has no bfloat16.
2. every dtype cast `__init__`, load point cloud and save point cloud make must be lossless: it never changes a value, in the mathematical sense. a cast that would change one hard-asserts and the program aborts.
   1. each dtype is a set of values, and one dtype's set may sit inside another's. float32's sits inside float64's. every casting decision reads those sets and the values a field holds, never the dtype names alone.
   2. when a system lacks a conceptual dtype but has one whose set contains its entire set, the smallest such dtype is used, and the cast converts whichever values are present in the data.
      1. in torch storage, ply u2 and numpy uint16 both go to int32, and ply u4 and numpy uint32 both go to int64.
      2. in a ply column, a bool target goes to u1: i1 and u1 are both one byte and both contain bool's two values, and u1 is the one whose signedness matches bool's.
   3. when the system has no such dtype, the largest narrower one it supports is used and no smaller dtype is considered after it, and the values then decide. every value inside that dtype's set means nothing is lost, so the cast converts. any value outside means something is lost, so the cast hard-asserts and the program aborts.
      1. in torch storage, a float128 source with no override uses float64. float32 and smaller dtypes are not considered.
      2. in a ply column, an int64 target goes to i4 and a uint64 target goes to u4.
   4. no field name changes the decision. xyz, rgb, indices, feat, colors and normals cast by the same rules as any other field.
   5. a lossy cast belongs to the caller of these modules and never to the modules themselves. a caller wanting float32 coordinates out of a float64 source narrows them itself and hands the narrowed values in.
3. determining the dtype from the source, one rule per source:
   1. an in-memory variable defines the dtype its tensor or array carries.
   2. a .pth defines the dtype the stored tensor or array carries.
   3. a .ply defines each column's stored dtype character, so an f4 column defines float32 and an f8 column defines float64, and it stores colors as u1.
   4. a .pcd defines the dtype each open3d attribute carries.
   5. a .las or .laz defines the dtype laspy materializes each dimension as: float64 for the scaled x, y and z, uint16 for the colors, and uint8 for a bit-packed field, which is an ordinary unsigned integer.
   6. a .txt holds decimal text, which yields float64.
   7. a .off holds decimal text, and float32 is what load point cloud keeps it at, hard-asserting on any magnitude float32 cannot hold rather than moving it onto a dtype that covers it.

#### 1.1.2. Color Data Convention Conversion

1. color conventions: rgb admits any integer dtype and any float dtype, unlike mesh vertex colors. conventions include:
   1. 0 to 255 unsigned integer representation.
   2. -128 to 127 signed integer representation.
   3. 0 to 1 floating point representation.
   4. 0 to 65535 unsigned integer representation.
2. conversion between conventions:
   1. range mapping: for source range $[a, b]$ and target range $[c, d]$, each channel value $x$ maps to $y = c + (x - a)(d - c)/(b - a)$ before rounding.
      1. 0 to 255 into 0 to 1: $y = x/255$.
      2. -128 to 127 into 0 to 1: $y = (x + 128)/255$.
      3. -128 to 127 into 0 to 255: $y = x + 128$.
      4. 0 to 65535 into 0 to 255: $y = x/257$.
      5. the reverse conversion uses the same formula with the source and target ranges exchanged.
   2. target representation:
      1. a floating point target uses $y$ without integer rounding.
      2. an integer target rounds $y$ to the nearest integer.
   3. losslessness: a conversion is lossless when the source values are exactly recoverable by converting the result back to the source convention, and lossy otherwise. the conversion performs either one, because tolerating the loss belongs to whoever asked for the target convention.
      1. 0 to 65535 into 0 to 255: a value of 1 rounds to 0 and converts back to 0, so the conversion is lossy.
      2. 0 to 65535 into 0 to 255: a value of 257 converts to 1 and back to 257, so the conversion is lossless.
3. naming conventions by dtype: the conventions are told apart by dtype and never by inspecting the values, the same way `validate_vertex_color` tells mesh vertex colors apart. the naming dtype is the data's own conceptual dtype, not the dtype of the tensor holding it, so uint16 color data held in an int32 tensor is named uint16. integer conventions span their dtype's full range.
   1. uint8 names the 0 to 255 convention.
   2. int8 names the -128 to 127 convention.
   3. a float dtype names the 0 to 1 convention.
   4. uint16 names the 0 to 65535 convention.

#### 1.1.3. Layout Mapping

1. what it is: the mapping between the source layout and the loaded layout, with the columns the source held on one side and the fields assembled from them on the other.
2. forward mapping: determining the layout from the source, one rule per source. each field carries the name its source gives the column, attribute or dimension it holds, and a caller wanting a field under another name, or assembled out of several columns, states that in the meta data.
   1. an in-memory variable uses the identity mapping: the name a field was handed under stands for the whole block of columns it was handed as.
   2. a .pth holds one block of unnamed columns and defines no column-to-field mapping. its columns are named by position.
   3. a .ply names each column, so a column called x becomes a field called x and a column called intensity becomes a field called intensity.
      1. in a file with more than one separately named group of columns, a column's name is its group's name and its own together, so a column called x in a group called vertex becomes a field called vertex.x.
   4. a .pcd names each open3d attribute, so the positions attribute becomes a field called positions and the colors attribute becomes a field called colors.
   5. a .las or .laz names each laspy dimension separately and as ply does.
   6. a .txt holds unnamed columns and defines no column-to-field mapping. its columns are named by position.
   7. a .off names no columns and defines no column-to-field mapping. the OFF format declares its vertex block to be the point data, and those columns are named by position.
3. reverse mapping: each field maps back to the source column names its mapping records. a field mapped from ('x', 'y', 'z') maps back to x, y and z.

#### 1.1.4. New Meta Data API

1. structure: it has two parts
   1. dtype:
      1. the record always keeps the source dtype.
      2. it records the conceptual dtype. a field entering as ply u2, as numpy uint16, or as an open3d UInt16 all record the same thing.
      3. it is recorded against the source layout and not the loaded layout: the dtype is the one the source column held, not the one the loaded field carries.
         1. for the ply u4 example in Type Casting, the record holds uint32.
         2. a float128 source records float128 in meta data.
   2. layout: it records the mapping defined by Layout Mapping.
2. granularity: the record is one whole, created when the obj is constructed. inside it, the dtype and the layout are both keyed on the source columns.
3. immutability: the record is never mutable. adding a field, deleting a field, and overwriting an existing field all leave it exactly as it was.
   1. user of PointCloud obj may however modify the fields, but the meta data stays constant and immutable once created.
4. types of meta data: there are four meta data: the recorded meta data, the override meta data, the default meta data, and the target meta data the other three resolve into.
   1. recorded meta data: meta data records what the source looked like, upon construction. it records the source of the data, wherever the data comes from: a load from disk, or a construction from a torch tensor or a numpy array. the meta data travels with the obj.
      1. Select preserves it.
      2. serializing a `PointCloud` and restoring it preserves it. a cache is not a source, so restoring builds no new record.
      3. constructing a `PointCloud` from another obj's fields inherits that obj's record. another obj is not a source, so construction builds no new record.
   2. override meta data:
      1. `__init__`, load point cloud and save point cloud each accept an override, and it reaches both the dtype and the layout at each.
   3. default meta data (default layout): each format's per-format helper defines the default layout for its own format, on load and on save alike.
      1. .ply and .las/.laz on load: when you see x, y, and z, default to stacking them into one field called xyz. when you see red, green, and blue, default to stacking them into one field called rgb.
      2. .ply on save: xyz splits back into x, y and z, and rgb splits back into red, green and blue.
      3. .pcd, .pth, .txt and .off: no default.
      4. no other defaults defined for now.
   4. target meta data:
      1. the target has one entry for each field the obj holds:
      2. definition:
         1. the entry takes whichever dtype and layout the override states.
         2. it takes the default layout wherever the override states none.
         3. it takes the recorded ones wherever neither the override nor the default states any.
         4. where the record names the field nowhere, the field supplies both: its name serves as the layout, and the dtype it carries as the dtype.
5. applying meta data:
   1. how the derived target is applied. the target resolves first, and the layout mapping is checked only after that:
      1. dtype (and convention):
         1. color conversion happens in two steps:
            1. where a conversion is defined for the pair, the values are mapped from the convention the current dtype names to the convention the target dtype names.
            2. after convention conversion, type casting happens normally.
         2. every other field goes through a direct type cast.
         3. lossless is asserted. the target is applied if it's lossless. the program hard asserts if lossless cannot be achieved.
      2. layout:
         1. the target's layout assembles the field from the columns it names
         2. if the columns a target layout merges into one field still hold different dtypes once the target dtype has been applied, the program hard asserts and aborts.
   3. applying the target changes the fields the obj stores and never the record, which stays exactly what the source data held.

#### 1.1.5. Point Cloud Data Structure Construction and I/O

1. the `PointCloud` class:
   1. common construction by `__init__` from in-memory variables or by load point cloud from files:
      1. no canonicalization: `PointCloud` does not canonicalize any field, color included.
         1. rgb enters and is held exactly as it arrived, like every other field.
         2. fields keep their own names.
      2. the ONLY place init may ever have any type casting ops is by invoking the `apply_meta_data`.
   2. validation:
      1. the columns a field is assembled from must all hold one dtype once the target dtype has been applied. disagreeing column dtypes hard-assert and abort rather than being promoted to a dtype covering them all.
      2. `PointCloud` keeps validating xyz and rgb by field name.
         1. xyz is any floating point dtype.
         2. `PointCloud` enforces that rgb values lie inside the range of their current color convention, as Color Data Convention Conversion defines it.
            1. a floating point rgb carrying a value outside 0 to 1 is refused. `PointCloud` hard-asserts and the program aborts, both when the field enters and on every later assignment to it.
   3. replacing rgb with a clone preserves its existing color convention.
2. load point cloud
   1. the per-format helpers
      1. they load and
         1. do necessary type casting when the dtype systems mismatch and when the type cast can be lossless.
         2. never change layout.
      2. they construct the meta data record from the data in disk, NOT from the type-casted data stored in the PointCloud obj. i.e., the recorded meta data is a consequence of what's inside the file in disk and nothing else.
      3. they define and apply the default layout for each format.
      4. the only silent cast is the one that resolves a dtype system mismatch, and nothing beyond it happens silently. any further lossless dtype change is the user's to instruct through the override.
   2. the main load API
      1. accepts a `meta_data` optional arg override.
      2. calls the `PointCloud.apply_meta_data` passing down the `meta_data` optional arg after the per-format helpers return.
      3. it knows nothing about default layout.
      4. never silently casts a dtype.
3. save point cloud
   1. the per-format helpers
      1. they save and
         1. do necessary type casting when dtype systems mismatch and when type cast can be lossless.
         2. never change layout
      2. meta data reaches a helper in no form at all: not the record, not the override, and not a target derived from either, whatever it is called. it's just completely unrelated to the job of the per-format helpers.
   2. the main save API
      1. accepts a `meta_data` optional arg override.
      2. calls the `PointCloud.apply_meta_data` passing down the `meta_data` optional arg. e.g.:
         1. turns each multi-column field back into one field per output column, each named by the reverse mapping, so a cloud holding xyz becomes fields x, y and z.
      3. passes that cloud to the per-format helpers.
4. consumers/users of `PointCloud`:
   1. any consumer of PointCloud in Pylon should be adjusted to work with the new design of PointCloud and its I/O.
      1. every caller passing dtype is updated to the meta data override.
      2. Select asserts that indices are int64 at the point of use.
      3. the point cloud displays under `data/viewer/utils/displays/points/dash` and `data/viewer/utils/displays/points/ts` assume 0 to 255 colors, and each applies Color Data Convention Conversion to rgb in its input normalization.

#### 1.1.6. What Becomes Stale Design

- the color rescale that guesses a [0, 1] range from the values and multiplies by 255
- the narrowing of every integer field to i4
- writing xyz as f4 whatever its dtype
- the _seg filename test that casts feat to int64
- PointCloud requiring indices to be int64
- retired layout behavior:
   1. the colors and pos aliases.
   2. splitting unnamed columns by position:
      1. loading a .pth taking columns zero through two as xyz and every column past the third as feat.
      2. loading a .txt taking column six alone as feat when the file holds seven or more columns and every column past the third otherwise.
   3. deriving x, y, z and red, green, blue from the field name, and the feat_0, feat_1 suffix fallback for anything else.
   4. load point cloud naming a field xyz or rgb from the columns it read, for any source names beyond the x, y, z and red, green, blue the default meta data covers.
- retired load point cloud arguments:
   1. the meta data override replaces the existing dtype arg, which cast xyz alone, and controls dtype per field.
   2. name_feat is removed, and the meta data override covers the dtype it formerly forced.
      1. name_feat's renaming of a named column to feat and its reshape to [N, 1] are dropped rather than replaced because of the field-name preservation required by Point Cloud Data Structure Construction and I/O.
   3. nameInPly is removed.

#### 1.1.7. Seriously Bad Behavior Observed when Working on this Task

The following are mistakes repeated again and again and every time when i asked what's unclear the agent tells me it's clear enough. I hate this behavior. The following mistakes are recorded here and persisted to let you see how bad you have been behaving. this is a explicitly and strictly and permanently banned.

1. save ply function taking a new arg called "target".
2. an additional argument called `layout` or `dtype` beside `meta_data` on `__init__`, load point cloud or save point cloud that's meant to do what `meta_data` is expected to cover.
3. adding a new method to `PointCloud` being `conceptual_dtype`.

### 1.2. Solution Constraints

1. You must use "meta_data" as the name of the new arg of init, load, and save. nothing else accepted. it is the only new arg any of the three takes: whatever else a design wants to pass fits inside `meta_data` or is derived, and no second arg is added beside it.
2. `PointCloud` should expose a public method `apply_meta_data`, which also takes a `meta_data` arg as override.
   1. `apply_meta_data` must have a local helper that infers/derives the meta data from the recorded meta data and checks the override meta data to produce the target meta data.
   2. `apply_meta_data` must have a local helper that applies the target meta data to self.
3. `PointCloud.__init__`
   1. must have a local function to build meta data from provided source data and set class attr.
   2. must use `self.apply_meta_data` to apply meta data, giving it the `meta_data` the constructor was handed as the override.

## 2. Definition of Done

1. Skeleton design and code conformance to skeleton both done. Tests in-scope all passes.
2. This branch is rebased onto latest `main`.
3. Confirmation message that this task is all done and this branch is good to merge.

### 2.1. Project Consumers be Refactored

consumers are equivalently refactored: what a consumer does is what it did before, and only the API it reaches PointCloud through changes.

This commit "[Project][Tasks] Merge 20260903_integrate_blend_texture_not_render (#17)" in the iVISION project made a patch to `data/structures/three_d/point_cloud/io/load_point_cloud.py` to silence the dtype bug with point clouds. This task should be considered as the official solution to be adopted. Once this task's branch is merged into `Pylon:main`, the iVISION project should have their main rebased onto `Pylon:main` (a mirror `lib` in the iVISION project), so that the patch to `data/structures/three_d/point_cloud/io/load_point_cloud.py` is discarded from that commit and the new design by this task is adopted in the iVISION project.

### 2.2. Task Scope

1. load: .pth, .ply, .pcd, .las, .laz, .off, .txt. save: .ply. neither expands.
2. constructing a `PointCloud` from numpy arrays is in scope. the obj always stores torch tensors.
3. uint64 is unsupported as a source dtype for `__init__` or load point cloud and as a dtype in their meta data overrides. either case hard-asserts and aborts, regardless of the actual values. an override requesting another dtype does not make a uint64 source acceptable.
4. complex and float128 are in scope, ruled in or out per case by the same representability test as every other dtype rather than by their names.
5. convention conversion is not avoidable: save point cloud does it, and so do the point cloud displays under `data/viewer/utils/displays/points`, each reading its conventions off a dtype. what is out of scope is the effort of building a general named-convention mechanism with conversions between named conventions.
6. every consumer this change breaks is fixed within this task, together with its tests. merging a branch that leaves a consumer broken breaks main.
7. tests in scope are anything this task might possibly impact. that resolves to the 57 test files referencing `PointCloud`, its I/O or `Select`: the point cloud I/O suites, the `PointCloud` and `Select` suites, the vision-3d transform suites, the PCR collators and dataloaders, the viewer point cloud display suites, the PCR dataset suites, and the point cloud model and render suites.
