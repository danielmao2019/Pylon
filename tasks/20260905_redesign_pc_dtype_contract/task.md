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
  - [2.1. Task Scope](#21-task-scope)

----------

## 1. Guidelines

### 1.1. Proposed Solution

#### 1.1.1. Type Casting

1. the fundamental root cause is the dtype system mismatch. torch has no uint16 and no uint32, so ply u2 loads as int32 and ply u4 loads as int64. everything else loads unchanged: i1 as int8, u1 as uint8, i2 as int16, i4 as int32, f4 as float32, f8 as float64, b1 as bool. numpy also has uint64 and float128 that torch 2.2.2 does not, torch has bfloat16 that numpy 1.26.4 does not, and ply has no 64-bit integer.
2. there is one universe of conceptual dtypes, system-agnostic. every concrete system supports a subset of it, and no system's subset contains every other's.
3. the core principles of lossless dtype casting:
   1. a dtype is a set of values, and one dtype's set may sit inside another's. float32's sits inside float64's.
   2. convert or refuse is decided by whether the data falls inside the target dtype's set, never by the pair of dtype names. inside means nothing is lost, so convert. outside means something is lost, so refuse.
   3. no dtype is ruled out in advance: complex, uint64 and float128 are each handled by this same test.
   4. dtype system mismatch: no field name is special. xyz, rgb, indices, feat, colors, normals are ordinary fields. converting uint16 to int32 because torch lacks uint16 is a patch for that mismatch, not a color convention conversion.
   5. for both `__init__` and load point cloud, when torch does not support the target dtype, storage uses the smallest torch dtype whose set contains the target dtype's set, if one exists. ply u2 and numpy uint16 both go to int32, ply u4 and numpy uint32 both go to int64.
      1. if no containing dtype exists, test the largest narrower dtype supported by torch. convert only if the actual values are exactly representable in it; otherwise hard-assert and abort. do not test progressively smaller dtypes.
      2. a float128 source with no override records float128 in meta data and uses float64 storage if its actual values fit exactly. float32 and smaller dtypes are not tested.

#### 1.1.2. Color Data Convention Conversion

      3. color representation and convention conversion:
         1. rgb is either
            1. an integer dtype, where the dtype's own range is the colour range, or
            2. a floating point dtype of any width holding 0 to 1.
            3. the two are told apart by dtype and never by inspecting the values, the same way `validate_vertex_color` tells mesh vertex colours apart.
            4. it differs from that mesh rule in admitting any integer width and any float width, because ply stores colours as u1 while las stores them as uint16, and a reader neither widens nor narrows what the file holds.
         2. rgb enters and is held exactly as it arrived, like every other field.
         3. color convention: conversion changes between color representations, such as 0 to 255 integer representation and 0 to 1 floating point representation. save's color convention conversion is keyed on rgb and is the only such branch in the I/O layer.
         4. rgb is the one field with convention conversion between color representations. save reads the convention off the field's current dtype and off the recorded dtype, and converts between the two. the dtype defines the convention: a float dtype means 0 to 1, an integer dtype means that dtype's own range. so a uint8 color is 0 to 255 and a uint16 color is 0 to 65535, which is what las stores.
            1. the conversion rounds, and that rounding is lossless as this design defines loss. a colour that arrived from an integer source sits exactly on that range's grid, so it converts back to the value it came from.
            2. what save asserts is that the values sit inside the range the field's current dtype declares: 0 to 1 for a float, the dtype's own range for an integer.
            3. a colour that does not sit on the target grid was put there by the user modifying the field, and the rounding it then takes is the user's own concern. the module rounds and does not refuse.
#### 1.1.3. New Meta Data API

      2. new attr: meta data.
         1. what it is: meta data records what the source looked like, upon construction. it records the source of the data, wherever the data comes from: a load from disk, a construction from a torch tensor or a numpy array, or addition or deletion of fields. it records two things.
            1. dtype:
               1. the record always keeps the source dtype.
               2. it records the conceptual dtype. a field entering as ply u2, as numpy uint16, or as an open3d UInt16 all record the same thing.
               3. it is recorded against the source layout and not the loaded layout: the dtype is the one the source column held, not the one the loaded field carries. a ply u4 column loads as int64 because torch has no uint32, the record holds uint32, and save writes it back as u4.
               4. a las bit-packed field is an ordinary unsigned integer. laspy materializes it as uint8, so uint8 is what enters the obj and uint8 is what the record stores. no special treatment.
            2. layout:
               1. it records the mapping between the source layout and the loaded layout: the columns the source held on one side, the fields the reader assembled them into on the other. a ply maps ('x', 'y', 'z') to xyz, maps ('red', 'green', 'blue') to rgb, and maps ('intensity',) to intensity.
               2. a field that never came from a file records no layout, because it had no source columns.
         2. granularity: the record is per-field, created when a field enters the obj and deleted when the field is removed. inside a field, both halves are keyed on the source columns.
         3. immutability: for each field, the record is never mutable. an overwrite of a field that already exists must NOT change the meta data.
            1. user of PointCloud obj may however modify the fields, but the meta data stays constant and immutable once created.
         4. the meta data travels with the field, so Select preserves it.
         5. for `__init__` and load point cloud, the target dtype is the source dtype unless a meta data override specifies another dtype.
         6. `__init__` accepts an optional meta data override for construction from in-memory variables, just as load point cloud does. a dtype override changes the target dtype without changing the source dtype recorded in meta data.
         7. load point cloud should take an optional arg to override the meta data, the same way save point cloud does, and it reaches both halves.
            1. the dtype half changes only the value handed back. the record stays the dtype the source column held.
            2. the layout half chooses which source columns the reader assembles together. that is the loaded side of the mapping, so the record's loaded side is what the override asked for while its source side stays the columns the file held.
            3. the new API on the meta data override:
               1. this override replaces the existing dtype arg, which cast xyz alone. it is the same control at per-field granularity, so every caller passing dtype is updated.
               2. name_feat is removed. the override covers the dtype it forced, while its renaming of a named column to feat and its reshape to [N, 1] are dropped rather than replaced, because fields keep their own names and no caller outside a test passes it.
         8. save point cloud: meta data defaults to that version in the PointCloud obj, but overridable by an optional arg to control how it wants the field to be saved as.
            1. the override arg overrides either half of the record.

#### 1.1.4. Point Cloud Data Structure Construction and I/O

4. The core design change in this task:
   1. the `PointCloud` class:
      1. common construction by `__init__` from in-memory variables or by load point cloud from files:
         1. no canonicalization: `PointCloud` does not canonicalize any field, color included.
         2. validation:
            1. `PointCloud` keeps validating xyz and rgb by field name.
            2. xyz is any floating point dtype.
   2. consumers/users of `PointCloud`:
      1. any consumer of PointCloud in Pylon should be adjusted to work with the new design of PointCloud and its I/O.
      2. point cloud I/O:
         1. load point cloud
   6. load point cloud preserves everything whenever possible, and converts dtype only for the mismatch between torch and the format it is reading.
            1. a reader never widens a field it builds: the record keeps the dtype the file stores each coordinate column in, so an f4 ply gives float32 xyz and an f8 ply gives float64 xyz.
            2. the columns a field is assembled from must all hold one dtype. the reader hard-asserts it, and a file whose columns disagree aborts the program rather than being promoted to a dtype covering them all.
            4. the .off reader keeps building float32 and hard-asserts it is never handed anything beyond what it can already handle, rather than widening to cover it.
         2. save point cloud
            1. strictly follows the meta data. it does not need to be aware of the dtype mismatch at all.
            2. save point cloud recovers both halves of the record:
               1. dtype: save casts each column of a field to the dtype recorded for that source column.
                  1. defensive programming: it asserts the cast is lossless. a target dtype ply does not have, such as int64, is never substituted for. save hard-asserts and the program aborts. with no override in place the target is the record, so this is the user asking for int64 in a ply file, and it is the user's own doing.
               2. layout: save writes each column of a field under the source column name the mapping gives it, instead of deriving names from the field name. a mapping that does not name exactly as many source columns as the field has columns leaves save with no names to write under, so it is refused unless the override supplies them. a field with no recorded layout is that same case.
#### 1.1.5. what becomes stale design

- the color rescale that guesses a [0, 1] range from the values and multiplies by 255
- the narrowing of every integer field to i4
- writing xyz as f4 whatever its dtype
- the _seg filename test that casts feat to int64
- the colors and pos aliases
- the writer deriving x, y, z and red, green, blue from the field name, and its feat_0, feat_1 suffix fallback for anything else
- PointCloud requiring indices to be int64. Select asserts it at the point of use instead

## 2. Definition of Done

1. Skeleton design and code conformance to skeleton both done. Tests in-scope all passes.
2. This branch is rebased onto latest `main`.
3. Confirmation message that this task is all done and this branch is good to merge.

### 2.1. Task Scope

1. load: .pth, .ply, .pcd, .las, .laz, .off, .txt. save: .ply. neither expands.
2. constructing a `PointCloud` from numpy arrays is in scope. the obj always stores torch tensors.
3. convention conversion is not avoidable and save point cloud does it, reading each convention off a dtype. what is out of scope is the effort of building a general named-convention mechanism with conversions between named conventions.
4. every consumer this change breaks is fixed within this task, together with its tests. merging a branch that leaves a consumer broken breaks main.
5. tests in scope are anything this task might possibly impact. that resolves to the 57 test files referencing `PointCloud`, its I/O or `Select`: the point cloud I/O suites, the `PointCloud` and `Select` suites, the vision-3d transform suites, the PCR collators and dataloaders, the viewer point cloud display suites, the PCR dataset suites, and the point cloud model and render suites.
