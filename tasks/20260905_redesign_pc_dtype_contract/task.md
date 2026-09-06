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
- [2. Definition of Done](#2-definition-of-done)
  - [2.1. Project Consumers be Refactored](#21-project-consumers-be-refactored)
  - [2.2. Task Scope](#22-task-scope)

----------

## 1. Guidelines

### 1.1. Proposed Solution

#### 1.1.1. Type Casting

1. the fundamental root cause is the dtype system mismatch: the dtype systems are each a subset of one universal, system-agnostic collection of conceptual dtypes, and no system's subset contains every other's.
   1. conceptual dtype identity across systems:
      1. uint16 and int32 are two distinct conceptual dtypes; numpy int32 and torch int32 represent the same conceptual dtype.
      2. every ply dtype torch carries loads unchanged: i1 as int8, u1 as uint8, i2 as int16, i4 as int32, f4 as float32, f8 as float64, b1 as bool.
   2. each system's supported subset:
      1. ply's subset is b1, i1, u1, i2, u2, i4, u4, f4 and f8, so ply has no 64-bit integer.
      2. torch 2.2.2 has no uint16, uint32, uint64 or float128, and bfloat16 is its alone.
      3. numpy 1.26.4 has uint64 and float128, and has no bfloat16.
2. every dtype cast `__init__`, load point cloud and save point cloud perform must be lossless: it never changes a value, in the mathematical sense.
   1. each dtype is a set of values, and one dtype's set may sit inside another's. float32's sits inside float64's. every casting decision reads those sets and the values a field holds, never the dtype names alone.
   2. when a system lacks a conceptual dtype but has one whose set contains its entire set, the smallest such dtype is used, and the cast converts whichever values are present in the data.
      1. in torch storage, ply u2 and numpy uint16 both go to int32, and ply u4 and numpy uint32 both go to int64.
   3. when the system has no such dtype, the largest narrower one it supports is used and no smaller dtype is considered after it, and the values then decide: every value inside that dtype's set means nothing is lost, so the cast converts; any value outside means something is lost, so the cast hard-asserts and the program aborts.
      1. in torch storage, a float128 source with no override uses float64. float32 and smaller dtypes are not considered.
      2. in a ply column, an int64 target goes to i4 and a uint64 target goes to u4.
   4. no field name changes the decision. xyz, rgb, indices, feat, colors and normals cast by the same rules as any other field.

#### 1.1.2. Color Data Convention Conversion

1. color conventions: rgb admits any integer width and any float width, unlike mesh vertex colors. conventions include:
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
   3. losslessness: the conversion proceeds only if the source values are exactly recoverable by converting the result back to the source convention; otherwise it hard-asserts and aborts.
      1. 0 to 65535 into 0 to 255: a value of 1 rounds to 0 and converts back to 0, so the conversion is lossy.
      2. 0 to 65535 into 0 to 255: a value of 257 converts to 1 and back to 257, so the conversion is lossless.
3. naming conventions by dtype: the conventions are told apart by dtype and never by inspecting the values, the same way `validate_vertex_color` tells mesh vertex colors apart. integer conventions span their dtype's full range.
   1. uint8 names the 0 to 255 convention.
   2. int8 names the -128 to 127 convention.
   3. a float dtype names the 0 to 1 convention.
   4. uint16 names the 0 to 65535 convention.

#### 1.1.3. Layout Mapping

#### 1.1.4. New Meta Data API

1. what it is: meta data records what the source looked like, upon construction. it records the source of the data, wherever the data comes from: a load from disk, a construction from a torch tensor or a numpy array, or addition or deletion of fields. it records two things.
   1. dtype:
      1. the record always keeps the source dtype.
      2. it records the conceptual dtype. a field entering as ply u2, as numpy uint16, or as an open3d UInt16 all record the same thing.
      3. it is recorded against the source layout and not the loaded layout: the dtype is the one the source column held, not the one the loaded field carries.
         1. for the ply u4 example in Type Casting, the record holds uint32.
         2. a float128 source with no override records float128 in meta data.
   2. layout:
      1. it records the mapping between the source layout and the loaded layout: the columns the source held on one side, the fields the reader assembled them into on the other. a ply maps ('x', 'y', 'z') to xyz, maps ('red', 'green', 'blue') to rgb, and maps ('intensity',) to intensity.
      2. a field constructed from an in-memory variable records the identity mapping: the name it was handed under stands for the whole block of columns it was handed as.
2. granularity: the record is per-field, created when a field enters the obj and deleted when the field is removed. inside a field, both halves are keyed on the source columns.
3. immutability: for each field, the record is never mutable. an overwrite of a field that already exists must NOT change the meta data.
   1. user of PointCloud obj may however modify the fields, but the meta data stays constant and immutable once created.
4. the meta data travels with the field.
   1. Select preserves it.
   2. serializing a `PointCloud` and restoring it preserves it. a cache is not a source, so restoring builds no new record.
5. for `__init__` and load point cloud, the target dtype is the source dtype.
6. the meta data override:
   1. `__init__`, load point cloud and save point cloud each accept one, and it reaches both halves at each.
   2. it is optional where the source defines a half, and it replaces that half when it states one.
   3. it is required where the source does not define a half: the caller supplies that half, and a construction or load without it hard-asserts and aborts.
      1. a .pth holds one block of columns the file names nothing, so it defines no layout and the caller's override names the columns.
      2. a .txt holds columns the file names nothing, so it defines no layout and the caller's override names the columns.
      3. a ply carrying more than one element does not define which element's columns a field is assembled from, so the caller's override names them.
   4. a dtype override changes the target dtype without changing the source dtype the record keeps.
   5. a layout override chooses which source columns are assembled into a field. the record's loaded side is what the override asked for, while its source side stays the columns the source held.
7. save point cloud: each field is written under the dtype and layout its record names, and save derives nothing else.
   1. the record is the target, including when it records int64 for a ply save.
   2. dtype: the dtype recorded for each source column is its save target. the actual ply storage dtype follows the lossless casting rule in Type Casting.
      1. the ply u4 example is therefore saved as u4.
   3. layout: save writes each field back out under the names the record maps it from. a field the record maps from ('x', 'y', 'z') is written out as x, y and z.

#### 1.1.5. Point Cloud Data Structure Construction and I/O

1. the `PointCloud` class:
   1. common construction by `__init__` from in-memory variables or by load point cloud from files:
      1. no canonicalization: `PointCloud` does not canonicalize any field, color included.
         1. rgb enters and is held exactly as it arrived, like every other field.
         2. fields keep their own names.
      2. both `__init__` and load point cloud apply Type Casting to the target dtype supplied by New Meta Data API.
         1. uint16 color data stored in an int32 tensor retains the 0 to 65535 color representation.
   2. validation:
      1. `PointCloud` keeps validating xyz and rgb by field name.
      2. xyz is any floating point dtype.
      3. `PointCloud` enforces that rgb values lie inside the range of their current color convention, as Color Data Convention Conversion defines it.
         1. a floating point rgb carrying a value outside 0 to 1 is refused. `PointCloud` hard-asserts and the program aborts, both when the field enters and on every later assignment to it.
   3. replacing rgb with a clone preserves its existing color convention.
2. consumers/users of `PointCloud`:
   1. any consumer of PointCloud in Pylon should be adjusted to work with the new design of PointCloud and its I/O.
      1. every caller passing dtype is updated to the meta data override.
      2. Select asserts that indices are int64 at the point of use.
      3. the point cloud displays under `data/viewer/utils/displays/points/dash` and `data/viewer/utils/displays/points/ts` assume 0 to 255 colors, and each applies Color Data Convention Conversion to rgb in its input normalization.
   2. point cloud I/O:
      1. load point cloud
         1. load point cloud preserves everything whenever possible, and applies Type Casting only for the dtype mismatch between torch and the format it is reading.
            1. a reader neither widens nor narrows what the file holds.
            2. a reader never widens a field it builds.
               1. an f4 ply gives float32 xyz and an f8 ply gives float64 xyz.
         2. the columns a field is assembled from must all hold one dtype. the reader hard-asserts it, and a file whose columns disagree aborts the program rather than being promoted to a dtype covering them all.
         3. the .off reader keeps building float32 and hard-asserts it is never handed anything beyond what it can already handle, rather than widening to cover it.
      2. save point cloud
         1. strictly follows the meta data. it does not need to be aware of the dtype mismatch at all.
            1. save point cloud recovers both halves of the record as specified by New Meta Data API, and its dtype casts follow Type Casting.
         2. save's color convention conversion is keyed on rgb and is the only such branch in the I/O layer.
            1. rgb is the one field with convention conversion between color representations.
            2. save applies Color Data Convention Conversion from the field's current color convention to the convention defined by the target conceptual dtype.
3. .las and .laz:
   1. coordinates are the scaled x, y and z. laspy materializes them as float64, so float64 is what enters the obj and what the meta data records.
   2. colors are stored as uint16, while ply stores colors as u1.
   3. a bit-packed field is an ordinary unsigned integer. laspy materializes it as uint8, so uint8 is what enters the obj and what the meta data records.

#### 1.1.6. What Becomes Stale Design

- the color rescale that guesses a [0, 1] range from the values and multiplies by 255
- the narrowing of every integer field to i4
- writing xyz as f4 whatever its dtype
- the _seg filename test that casts feat to int64
- the colors and pos aliases
- the readers splitting a block the file names nothing by column index: the .pth reader taking columns zero through two as xyz and every column past the third as feat, and the .txt reader taking column six alone as feat when the file holds seven or more columns and every column past the third otherwise
- the writer deriving x, y, z and red, green, blue from the field name, and its feat_0, feat_1 suffix fallback for anything else
- PointCloud requiring indices to be int64
- retired load point cloud arguments:
   1. the meta data override replaces the existing dtype arg, which cast xyz alone, and controls dtype per field.
   2. name_feat is removed, and the meta data override covers the dtype it formerly forced.
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
3. uint64 is unsupported as a source dtype for `__init__` or load point cloud and as a dtype in their meta data overrides. either case hard-asserts and aborts, regardless of the actual values. an override requesting another dtype does not make a uint64 source acceptable.
4. complex and float128 are in scope, ruled in or out per case by the same representability test as every other dtype rather than by their names.
5. convention conversion is not avoidable: save point cloud does it, and so do the point cloud displays under `data/viewer/utils/displays/points`, each reading its conventions off a dtype. what is out of scope is the effort of building a general named-convention mechanism with conversions between named conventions.
6. every consumer this change breaks is fixed within this task, together with its tests. merging a branch that leaves a consumer broken breaks main.
7. tests in scope are anything this task might possibly impact. that resolves to the 57 test files referencing `PointCloud`, its I/O or `Select`: the point cloud I/O suites, the `PointCloud` and `Select` suites, the vision-3d transform suites, the PCR collators and dataloaders, the viewer point cloud display suites, the PCR dataset suites, and the point cloud model and render suites.
