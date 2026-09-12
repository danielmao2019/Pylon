goal: re-design pc dtype contract/provenance

## Table of Contents <!-- omit in toc -->

- [1. Knowledge](#1-knowledge)
- [2. Guidelines](#2-guidelines)
  - [2.1. Problem Definition](#21-problem-definition)
  - [2.2. Proposed Solution](#22-proposed-solution)
    - [2.2.1. The new meta data tool](#221-the-new-meta-data-tool)
    - [2.2.2. Modules](#222-modules)
      - [2.2.2.1. build source meta data](#2221-build-source-meta-data)
      - [2.2.2.2. lossless dtype casting](#2222-lossless-dtype-casting)
      - [2.2.2.3. numpy to torch or torch to numpy](#2223-numpy-to-torch-or-torch-to-numpy)
      - [2.2.2.4. Color Data Convention Conversion](#2224-color-data-convention-conversion)
      - [2.2.2.5. apply target meta data](#2225-apply-target-meta-data)
    - [2.2.3. The Core Design](#223-the-core-design)
  - [2.3. Proposed Solution](#23-proposed-solution)
    - [2.3.1. Layout Mapping](#231-layout-mapping)
    - [2.3.2. Consumers](#232-consumers)
    - [2.3.3. What Becomes Stale Design](#233-what-becomes-stale-design)
    - [2.3.4. Seriously Bad Behavior Observed when Working on this Task](#234-seriously-bad-behavior-observed-when-working-on-this-task)
  - [2.4. Solution Constraints](#24-solution-constraints)
- [3. Definition of Done](#3-definition-of-done)
  - [3.1. Project Consumers be Refactored](#31-project-consumers-be-refactored)
  - [3.2. Task Scope](#32-task-scope)

----------

## 1. Knowledge

Any non-pth format can only deal with numpy arrays. If this library stores data in torch Tensors then there's a layer between numpy and torch.

For pth format it can also work with torch directly.

torch 2.2.2 has no uint16, uint32, uint64 or float128, and bfloat16, complex32, float8_e4m3fn and float8_e5m2 are its alone.

numpy 1.26.4 has uint64 and float128, and has no bfloat16.

numpy holds every dtype a .ply, .pcd, .las, .laz, .txt or .off file defines.

ply's subset is i1, u1, i2, u2, i4, u4, f4 and f8, so ply has no 64-bit integer and no boolean.

## 2. Guidelines

### 2.1. Problem Definition

1. A load-save round trip must preserve the values, dtypes, and layout strictly.
2. When user provides override meta data to init, load, or save, the system must either realize that strictly or fail fast and loud, and never deliver anything that's not what asked for.
3. `PointCloud` should store torch Tensors, never numpy arrays.
4. `PointCloud` should always store point positions as `xyz` field and point colors as `rgb` field.

### 2.2. Proposed Solution

#### 2.2.1. The new meta data tool

1. structure: it has two parts
   1. dtype: the conceptual dtype.
   2. layout: the mapping defined by Layout Mapping.
2. the four types of meta data: source meta data, default meta data, override meta data, and target meta data.
   1. source meta data: one whole record of what the source looked like, created when the obj is constructed, wherever the data comes from: a load from disk, or a construction from a torch tensor or a numpy array.
      1. content: the dtype part alone, keyed on the source columns.
         1. the dtype is the conceptual dtype of the raw data (not the loaded data).
         2. examples:
            1. for the ply u4 example in Type Casting, the record holds uint32.
            2. a float128 source records float128 in meta data.
            3. a field entering as ply u2, as numpy uint16, or as an open3d UInt16 all record the same thing.
      2. immutability:
         1. the record is never mutable, regardless of how the user of a `PointCloud` obj may modify its fields.
         2. examples: adding a field, deleting a field, and overwriting an existing field all leave it exactly as it was.
      3. travel: the record travels with the obj.
         1. Select preserves it.
         2. serializing a `PointCloud` and restoring it preserves it. a cache is not a source, so restoring builds no new record.
         3. constructing a `PointCloud` from another obj's fields inherits that obj's record. another obj is not a source, so construction builds no new record.
   3. default meta data (default layout): each format's per-format helper defines the default layout for its own format, on load and on save alike.
      1. .ply and .las/.laz on load: when you see x, y, and z, default to stacking them into one field called xyz. when you see red, green, and blue, default to stacking them into one field called rgb.
      2. .ply on save: xyz splits back into x, y and z, and rgb splits back into red, green and blue.
      3. .pcd, .pth, .txt and .off: default is to map the columns to '0', '1', '2', ...
      4. no other defaults defined for now.
   4. override meta data:
      1. init, load, and save each accept an override, and it reaches both the dtype and the layout at each.
   5. target meta data: it holds one entry per field the override or the default names, each holding that field's layout and, where the override states one, its dtype.
      1. for the layout of load and save:
         1. when a layout moves columns into another field, the fields those columns came from are dropped.
         2. each field takes its layout from the first of these that states one: the override, and the default.
      2. for layout of init:
         1. override is the target, because there's no default for init. the target of `__init__` is the meta data `__init__` is handed.
      3. for the dtype of init, load, and save:
         1. each entry takes it from the override.

#### 2.2.2. Modules

##### 2.2.2.1. build source meta data

1. conceptual dtypes:
   1. the fundamental root cause is the dtype system mismatch between numpy and torch: each is a subset of one universal, system-agnostic collection of conceptual dtypes, and neither's subset contains the other's.
   2. uint16 and int32 are two distinct conceptual dtypes. numpy int32 and torch int32 represent the same conceptual dtype
2. build from the source, shared by numpy and torch.

##### 2.2.2.2. lossless dtype casting

1. the principle: every dtype cast in init, load, and save, including those cross-numpy-torch and those applying the resolved target dtype, must be lossless. i.e., it never changes a value, in the mathematical sense. a cast that would change one hard-asserts and the program aborts.
2. the mental model:
   1. each dtype is a set of values, and one dtype's set may sit inside another's.
   2. examples:
      1. float32's sits inside float64's.
3. how to do lossless dtype casting:
   1. every casting decision reads those sets and the values a field holds, never the dtype names alone.
   2. when a system lacks a conceptual dtype but has one whose set contains its entire set, the smallest such dtype is used, and the cast converts whichever values are present in the data.
   3. when the system has no such dtype, the largest narrower one it supports is used and no smaller dtype is considered after it, and the values then decide. every value inside that dtype's set means nothing is lost, so the cast converts. any value outside means something is lost, so the cast hard-asserts and the program aborts.
   4. no field name changes the decision. xyz, rgb, indices, feat and normals cast by the same rules as any other field.
   5. examples:
      1. in torch storage, numpy uint16 goes to int32 and numpy uint32 goes to int64
      2. in torch storage, a float128 source with no override uses float64. float32 and smaller dtypes are not considered.
      3. an in-memory variable defines the dtype its tensor or array carries.
      4. a .pth defines the dtype the stored tensor or array carries.
      5. a .ply defines each column's stored dtype character, so an f4 column defines float32 and an f8 column defines float64.
      6. a .pcd defines the dtype each open3d attribute carries.
      7. a .las or .laz defines the dtype laspy materializes each dimension as: float64 for the scaled x, y and z, uint16 for the colors, and uint8 for a bit-packed field, which is an ordinary unsigned integer.
      8. a .txt holds decimal text, which yields float64.
      9. a .off holds decimal text, and float32 is what load point cloud keeps it at, hard-asserting on any magnitude float32 cannot hold rather than moving it onto a dtype that covers it.
      10. a ply u2 column is held as int32 and a ply u4 column as int64.
      11. in a ply column, a bool target goes to u1: i1 and u1 are both one byte and both contain bool's two values, and u1 is the one whose signedness matches bool's.
      12. in a ply column, an int64 target goes to i4.
4. a lossy cast belongs to the caller of these modules and never to the modules themselves. a caller wanting float32 coordinates out of a float64 source narrows them itself and hands the narrowed values in.

##### 2.2.2.3. numpy to torch or torch to numpy

1. perform lossless dtype casting from one dtype system to another, using the module described above.
2. no color convention conversion.
3. no layout change.
4. examples:
   1. every ply dtype torch carries loads unchanged: i1 as int8, u1 as uint8, i2 as int16, i4 as int32, f4 as float32, f8 as float64.

##### 2.2.2.4. Color Data Convention Conversion

1. color conventions: rgb admits any integer dtype and any float dtype, unlike mesh vertex colors. the conventions are told apart by the dtype the data carries and never by inspecting the values, the same way `validate_vertex_color` tells mesh vertex colors apart. integer conventions span their dtype's full range. conventions include:
   1. uint8 names the 0 to 255 unsigned integer representation.
   2. int8 names the -128 to 127 signed integer representation.
   3. a float dtype names the 0 to 1 floating point representation.
   4. uint16 names the 0 to 65535 unsigned integer representation.
      1. an int32 color is in the uint16 convention.
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

##### 2.2.2.5. apply target meta data

1. there should be a `apply_meta_data` function under a `utils` submodule of the point cloud data structure module.
2. `apply_meta_data` takes a `meta_data` arg and applies exactly the meta data it is handed, resolving nothing itself.
3. `apply_meta_data` should work for both torch and numpy.
4. for what the resolved target meta data does not specify, don't touch. what this implies (incomplete list):
   1. fields keep their own names, except where the default or an override names them.
5. the dtype applies first, and the layout mapping is checked only after that
6. apply dtype:
   1. for non-rgb fields or columns
      1. if dtype cast is lossless, then do it.
      2. otherwise, hard assert.
   2. for rgb field:
      1. if source and target dtype pair is a defined convention conversion, then do convention conversion.
      2. otherwise, if dtype cast is lossless, then do it.
      3. otherwise, hard assert.
   3. no cross-numpy-torch should happen.
7. apply layout:
   1. the target's layout assembles the field from the columns it names.
   2. if the columns a target layout merges into one field still hold different dtypes once the target dtype has been applied, the program hard asserts and aborts. the columns a field is assembled from must all hold one dtype once the target dtype has been applied. disagreeing column dtypes hard-assert and abort rather than being promoted to a dtype covering them all.

#### 2.2.3. The Core Design

1. validation:
   1. xyz is any floating point dtype.
   2. rgb:
      1. `PointCloud` enforces that rgb values lie inside the range of their current color convention, as Color Data Convention Conversion defines it. `PointCloud` hard-asserts and the program aborts, both when the field enters and on every later assignment to it.
      2. examples:
         1. a floating point rgb carrying a value outside 0 to 1 is refused. 
2. init
   1. the received `meta_data` arg is treated as target meta data directly (there isn't a second thing to resolve together).
   2. if init from numpy, do the following in sequence:
      1. build source meta data.
      2. numpy to torch.
      3. apply target meta data.
      4. assign to instance attr.
   3. if init from torch, do the following in sequence:
      1. build source meta data.
      2. apply target meta data.
      3. assign to instance attr.
3. load
   1. the main load API
      1. accepts a `meta_data` optional arg override.
      2. passes the `meta_data` optional arg down to the per-format helper.
      3. it knows nothing about default layout.
      4. never silently casts a dtype.
   2. per-format load helpers
      1. the received `meta_data` arg is treated as override meta data that overrides the per-format defaults.
      2. they define the default layout for each format.
      3. non-pth formats and pth format with numpy storage do the following steps in sequence
         1. load as numpy, preserving values, dtypes, and layouts strictly.
         2. resolve target meta data from user-provided override and per-format default.
         3. construct `PointCloud` obj, passing raw data and target meta data as is. then the init op builds the source meta data and applies the target meta data.
      4. pth format with torch storage do the following steps in sequence
         1. load as torch, preserving values, dtypes, and layouts strictly.
         2. resolve target meta data from user-provided override and per-format default.
         3. construct `PointCloud` obj, passing raw data and target meta data as is. then the init op builds the source meta data and applies the target meta data.
4. save
   1. the main save API
      1. accepts a `meta_data` optional arg override.
      2. passes the point cloud and the `meta_data` optional arg down to the per-format helper.
   2. per-format save helpers
      1. the received `meta_data` arg is treated as override meta data that overrides the per-format defaults.
      2. they define the default layout for each format.
      3. non-pth formats and pth format with numpy storage do the following steps in sequence
         1. torch to numpy
         2. resolve target meta data from user-provided override and per-format default.
         3. apply target meta data.
         4. for what's not specified by target meta data, apply source meta data.
         5. save as file to disk.
      4. pth format with torch storage do the following steps in sequence
         1. resolve target meta data from user-provided override and per-format default.
         2. apply target meta data.
         3. for what's not specified by target meta data, apply source meta data.
         4. save as file to disk.

### 2.3. Proposed Solution

#### 2.3.1. Layout Mapping

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

#### 2.3.2. Consumers

1. consumers/users of `PointCloud`:
   1. any consumer of PointCloud in Pylon should be adjusted to work with the new design of PointCloud and its I/O.
      1. every caller passing dtype is updated to the meta data override.
      2. Select asserts that indices are int64 at the point of use.
      3. the point cloud displays under `data/viewer/utils/displays/points/dash` and `data/viewer/utils/displays/points/ts` assume 0 to 255 colors, and each applies Color Data Convention Conversion to rgb in its input normalization.

#### 2.3.3. What Becomes Stale Design

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

#### 2.3.4. Seriously Bad Behavior Observed when Working on this Task

The following are mistakes repeated again and again and every time when i asked what's unclear the agent tells me it's clear enough. I hate this behavior. The following mistakes are recorded here and persisted to let you see how bad you have been behaving. this is a explicitly and strictly and permanently banned.

1. save ply function taking a new arg called "target".
2. an additional argument called `layout` or `dtype` beside `meta_data` on `__init__`, load point cloud or save point cloud that's meant to do what `meta_data` is expected to cover.
3. adding a new method to `PointCloud` being `conceptual_dtype`.

### 2.4. Solution Constraints

1. You must use "meta_data" as the name of the new arg of init, load, and save. nothing else accepted. it is the only new arg any of the three takes: whatever else a design wants to pass fits inside `meta_data` or is derived, and no second arg is added beside it.

## 3. Definition of Done

1. Skeleton design and code conformance to skeleton both done. Tests in-scope all passes.
2. This branch is rebased onto latest `main`.
3. Confirmation message that this task is all done and this branch is good to merge.

### 3.1. Project Consumers be Refactored

consumers are equivalently refactored: what a consumer does is what it did before, and only the API it reaches PointCloud through changes.

This commit "[Project][Tasks] Merge 20260903_integrate_blend_texture_not_render (#17)" in the iVISION project made a patch to `data/structures/three_d/point_cloud/io/load_point_cloud.py` to silence the dtype bug with point clouds. This task should be considered as the official solution to be adopted. Once this task's branch is merged into `Pylon:main`, the iVISION project should have their main rebased onto `Pylon:main` (a mirror `lib` in the iVISION project), so that the patch to `data/structures/three_d/point_cloud/io/load_point_cloud.py` is discarded from that commit and the new design by this task is adopted in the iVISION project.

### 3.2. Task Scope

1. load: .pth, .ply, .pcd, .las, .laz, .off, .txt. save: .ply. neither expands.
2. constructing a `PointCloud` from numpy arrays is in scope. the obj always stores torch tensors.
3. uint64 is excluded from this task: it is unsupported as a source dtype for `__init__` or load point cloud and as a dtype in any meta data override. either case hard-asserts and aborts, regardless of the actual values. an override requesting another dtype does not make a uint64 source acceptable.
4. complex and float128 are in scope, ruled in or out per case by the same representability test as every other dtype rather than by their names.
5. convention conversion is not avoidable: save point cloud does it, and so do the point cloud displays under `data/viewer/utils/displays/points`, each reading its conventions off a dtype. what is out of scope is the effort of building a general named-convention mechanism with conversions between named conventions.
6. every consumer this change breaks is fixed within this task, together with its tests. merging a branch that leaves a consumer broken breaks main.
7. tests in scope are anything this task might possibly impact. that resolves to the 57 test files referencing `PointCloud`, its I/O or `Select`: the point cloud I/O suites, the `PointCloud` and `Select` suites, the vision-3d transform suites, the PCR collators and dataloaders, the viewer point cloud display suites, the PCR dataset suites, and the point cloud model and render suites.
