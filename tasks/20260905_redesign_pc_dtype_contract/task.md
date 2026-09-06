goal: re-design pc dtype contract/provenance

## Table of Contents <!-- omit in toc -->

- [1. Guidelines](#1-guidelines)
  - [1.1. Proposed Solution](#11-proposed-solution)
    - [1.1.1. Type Casting](#111-type-casting)
    - [1.1.2. Color Data Convention Conversion](#112-color-data-convention-conversion)
    - [1.1.3. New Meta Data API](#113-new-meta-data-api)
    - [1.1.4. Point Cloud Data Structure Construction and I/O](#114-point-cloud-data-structure-construction-and-io)
    - [1.1.5. what becomes stale design](#115-what-becomes-stale-design)
- [2. Definition of Done](#2-definition-of-done)
  - [2.1. Project Consumers be Refactored](#21-project-consumers-be-refactored)
  - [2.2. Task Scope](#22-task-scope)

----------

## 1. Guidelines

### 1.1. Proposed Solution

#### 1.1.1. Type Casting

1. the fundamental root cause is the dtype system mismatch. torch has no uint16 and no uint32, so ply u2 loads as int32 and ply u4 loads as int64. everything else loads unchanged: i1 as int8, u1 as uint8, i2 as int16, i4 as int32, f4 as float32, f8 as float64, b1 as bool. numpy also has uint64 and float128 that torch 2.2.2 does not, torch has bfloat16 that numpy 1.26.4 does not, and ply has no 64-bit integer.
2. there is one universe of conceptual dtypes, system-agnostic. every concrete system supports a subset of it, and no system's subset contains every other's.
3. the core principles of lossless dtype casting:
   1. a dtype is a set of values, and one dtype's set may sit inside another's. float32's sits inside float64's.
   2. convert or refuse is decided by whether the data falls inside the target dtype's set, never by the pair of dtype names. inside means nothing is lost, so convert. outside means something is lost, so refuse.
   3. complex and float128 are handled by this same representability test.
   4. dtype system mismatch: no field name is special. xyz, rgb, indices, feat, colors, normals are ordinary fields. converting uint16 to int32 because torch lacks uint16 is a patch for that mismatch, not a color convention conversion.
4. when torch does not support the target dtype, consider the torch dtypes whose value sets are supersets of the target dtype's entire value set. if any exist, storage uses the smallest such superset, regardless of which values happen to be present in the data. ply u2 and numpy uint16 both go to int32, ply u4 and numpy uint32 both go to int64.
   1. if no containing dtype exists, test the largest narrower dtype supported by torch. convert only if the actual values are exactly representable in it; otherwise hard-assert and abort. do not test progressively smaller dtypes.
   2. a float128 source with no override uses float64 storage if its actual values fit exactly. float32 and smaller dtypes are not tested.
   3. a ply u4 column loads as int64 because torch has no uint32.
5. every dtype cast `__init__`, load point cloud and save point cloud perform is lossless: the cast never changes the value, in the mathematical sense. defensive programming asserts it.
   1. a target dtype ply does not have, including int64 and uint64, is cast at save to a dtype ply does have, and the actual values decide: save converts when they are exactly representable in that dtype, and hard-asserts and refuses to save when they are not. this holds whether the target comes from the record or from the save meta data override.
      1. the dtype tested is the largest narrower one ply has, and no progressively smaller dtype is tested after it. an int64 target is tested against i4, so an int64 field whose values all fit i4 is written as i4 and one whose values do not is refused.
      2. a uint64 target is tested against u4 under the same rule: values that all fit u4 are written as u4, and other values are refused.

#### 1.1.2. Color Data Convention Conversion

1. rgb is either
   1. an integer dtype, where the dtype's own range is the color range, or
   2. a floating point dtype of any width holding 0 to 1.
   3. the two are told apart by dtype and never by inspecting the values, the same way `validate_vertex_color` tells mesh vertex colors apart.
   4. it differs from that mesh rule in admitting any integer width and any float width, because ply stores colors as u1 while las stores them as uint16.
2. color convention: conversion changes between color representations, such as 0 to 255 integer representation and 0 to 1 floating point representation.
   1. the dtype defines the convention: a float dtype means 0 to 1, an integer dtype means that dtype's own range. so a uint8 color is 0 to 255 and a uint16 color is 0 to 65535, which is what las stores. the dtype that defines the convention is the conceptual dtype and never the torch dtype the field is stored in: a uint16 color held in an int32 tensor is a 0 to 65535 color, because int32 storage is the dtype system mismatch patch and not a convention change.
   2. the conversion rounds, and that rounding is lossless as this design defines loss. a color that arrived from an integer source sits exactly on that range's grid, so it converts back to the value it came from.
   3. what save asserts is that the values sit inside the range the field's current dtype declares: 0 to 1 for a float, the dtype's own range for an integer.
   4. a color that does not sit on the target grid was put there by the user modifying the field, and the rounding it then takes is the user's own concern. the module rounds and does not refuse.

#### 1.1.3. New Meta Data API

1. what it is: meta data records what the source looked like, upon construction. it records the source of the data, wherever the data comes from: a load from disk, a construction from a torch tensor or a numpy array, or addition or deletion of fields. it records two things.
   1. dtype:
      1. the record always keeps the source dtype.
      2. it records the conceptual dtype. a field entering as ply u2, as numpy uint16, or as an open3d UInt16 all record the same thing.
      3. it is recorded against the source layout and not the loaded layout: the dtype is the one the source column held, not the one the loaded field carries.
         1. for the ply u4 example in Type Casting, the record holds uint32.
         2. a float128 source with no override records float128 in meta data.
         3. for a las bit-packed field materialized as uint8, uint8 is what the record stores. no special treatment.
      4. the record keeps the dtype the file stores each coordinate column in.
   2. layout:
      1. it records the mapping between the source layout and the loaded layout: the columns the source held on one side, the fields the reader assembled them into on the other. a ply maps ('x', 'y', 'z') to xyz, maps ('red', 'green', 'blue') to rgb, and maps ('intensity',) to intensity.
      2. a field constructed from an in-memory variable records the identity mapping: the name it was handed under stands for the whole block of columns it was handed as.
2. granularity: the record is per-field, created when a field enters the obj and deleted when the field is removed. inside a field, both halves are keyed on the source columns.
3. immutability: for each field, the record is never mutable. an overwrite of a field that already exists must NOT change the meta data.
   1. user of PointCloud obj may however modify the fields, but the meta data stays constant and immutable once created.
4. the meta data travels with the field, so Select preserves it.
5. for `__init__` and load point cloud, the target dtype is the source dtype unless a meta data override specifies another dtype.
6. `__init__` accepts an optional meta data override for construction from in-memory variables, just as load point cloud does. a dtype override changes the target dtype without changing the source dtype recorded in meta data.
   1. the override reaches the layout half as well. the API allows it and the caller chooses whether to use it.
7. load point cloud should take an optional arg to override the meta data, the same way save point cloud does, and it reaches both halves.
   1. the dtype half changes only the value handed back. the record stays the dtype the source column held.
      1. the meta data override provides the same dtype control as the former dtype argument, at per-field granularity.
      2. the meta data override covers the dtype formerly forced by name_feat.
   2. the layout half chooses which source columns the reader assembles together. that is the loaded side of the mapping, so the record's loaded side is what the override asked for while its source side stays the columns the file held.
8. save point cloud: meta data defaults to that version in the PointCloud obj, but overridable by an optional arg to control how it wants the field to be saved as.
   1. the override arg overrides either half of the record.
      1. with no override in place, the target is the record, including when it records int64 for a ply save.
   2. dtype: each saved column of a field uses the dtype recorded for that source column.
      1. the ply u4 example is therefore saved as u4.
   3. layout: save writes each field back out under the names the record maps it from. a field the record maps from ('x', 'y', 'z') is written out as x, y and z.

#### 1.1.4. Point Cloud Data Structure Construction and I/O

1. the `PointCloud` class:
   1. common construction by `__init__` from in-memory variables or by load point cloud from files:
      1. no canonicalization: `PointCloud` does not canonicalize any field, color included.
         1. rgb enters and is held exactly as it arrived, like every other field.
         2. fields keep their own names.
      2. both `__init__` and load point cloud apply Type Casting to the target dtype supplied by New Meta Data API.
      3. validation:
         1. `PointCloud` keeps validating xyz and rgb by field name.
         2. xyz is any floating point dtype.
2. consumers/users of `PointCloud`:
   1. any consumer of PointCloud in Pylon should be adjusted to work with the new design of PointCloud and its I/O.
      1. every caller passing dtype is updated to the meta data override.
      2. Select asserts that indices are int64 at the point of use.
   2. point cloud I/O:
      1. load point cloud
         1. load point cloud preserves everything whenever possible, and applies Type Casting only for the dtype mismatch between torch and the format it is reading.
            1. a reader neither widens nor narrows what the file holds.
            2. a reader never widens a field it builds.
               1. an f4 ply gives float32 xyz and an f8 ply gives float64 xyz.
         2. the columns a field is assembled from must all hold one dtype. the reader hard-asserts it, and a file whose columns disagree aborts the program rather than being promoted to a dtype covering them all.
         3. a las bit-packed field is an ordinary unsigned integer. laspy materializes it as uint8, so uint8 is what enters the obj.
         4. the .off reader keeps building float32 and hard-asserts it is never handed anything beyond what it can already handle, rather than widening to cover it.
      2. save point cloud
         1. strictly follows the meta data. it does not need to be aware of the dtype mismatch at all.
            1. save point cloud recovers both halves of the record as specified by New Meta Data API; its dtype casts follow the saving-specific losslessness and unsupported-target checks in Type Casting.
         2. save's color convention conversion is keyed on rgb and is the only such branch in the I/O layer.
            1. rgb is the one field with convention conversion between color representations.
            2. save reads the conventions off the field's current dtype and off the recorded dtype, and applies Color Data Convention Conversion between the two.

#### 1.1.5. what becomes stale design

- the color rescale that guesses a [0, 1] range from the values and multiplies by 255
- the narrowing of every integer field to i4
- writing xyz as f4 whatever its dtype
- the _seg filename test that casts feat to int64
- the colors and pos aliases
- the writer deriving x, y, z and red, green, blue from the field name, and its feat_0, feat_1 suffix fallback for anything else
- PointCloud requiring indices to be int64
- retired load point cloud arguments:
   1. the meta data override replaces the existing dtype arg, which cast xyz alone.
   2. name_feat is removed.
      1. name_feat's renaming of a named column to feat and its reshape to [N, 1] are dropped rather than replaced because of the field-name preservation required by Point Cloud Data Structure Construction and I/O.
         1. another reason for dropping these two behaviors is that no caller outside a test passes name_feat.
   3. nameInPly is removed.

## 2. Definition of Done

1. Skeleton design and code conformance to skeleton both done. Tests in-scope all passes.
2. This branch is rebased onto latest `main`.
3. Confirmation message that this task is all done and this branch is good to merge.

### 2.1. Project Consumers be Refactored

This commit "[Project][Tasks] Merge 20260903_integrate_blend_texture_not_render (#17)" in the iVISION project made a patch to `data/structures/three_d/point_cloud/io/load_point_cloud.py` to silence the dtype bug with point clouds. This task should be considered as the official solution to be adopted. Once this task's branch is merged into `Pylon:main`, the iVISION project should have their main rebased onto `Pylon:main` (a mirror `lib` in the iVISION project), so that the patch to `data/structures/three_d/point_cloud/io/load_point_cloud.py` is discarded from that commit and the new design by this task is adopted in the iVISION project.

### 2.2. Task Scope

1. load: .pth, .ply, .pcd, .las, .laz, .off, .txt. save: .ply. neither expands.
2. constructing a `PointCloud` from numpy arrays is in scope. the obj always stores torch tensors.
3. convention conversion is not avoidable and save point cloud does it, reading each convention off a dtype. what is out of scope is the effort of building a general named-convention mechanism with conversions between named conventions.
4. every consumer this change breaks is fixed within this task, together with its tests. merging a branch that leaves a consumer broken breaks main.
5. tests in scope are anything this task might possibly impact. that resolves to the 57 test files referencing `PointCloud`, its I/O or `Select`: the point cloud I/O suites, the `PointCloud` and `Select` suites, the vision-3d transform suites, the PCR collators and dataloaders, the viewer point cloud display suites, the PCR dataset suites, and the point cloud model and render suites.
