goal: re-design pc dtype contract/provenance

## Table of Contents <!-- omit in toc -->

- [1. Guidelines](#1-guidelines)
  - [1.1. Proposed Solution](#11-proposed-solution)
- [2. Definition of Done](#2-definition-of-done)
  - [2.1. Task Scope](#21-task-scope)

----------

## 1. Guidelines

### 1.1. Proposed Solution

1. the fundamental root cause is the dtype system mismatch. torch has no uint16 and no uint32, so ply u2 loads as int32 and ply u4 loads as int64. everything else loads unchanged: i1 as int8, u1 as uint8, i2 as int16, i4 as int32, f4 as float32, f8 as float64, b1 as bool. numpy also has uint64 and float128 that torch 2.2.2 does not, torch has bfloat16 that numpy 1.26.4 does not, and ply has no 64-bit integer.
2. there is one universe of conceptual dtypes, system-agnostic. every concrete system supports a subset of it, and no system's subset contains every other's.
3. the core principles:
   1. a dtype is a set of values, and one dtype's set may sit inside another's. float32's sits inside float64's.
   2. convert or refuse is decided by whether the data falls inside the target dtype's set, never by the pair of dtype names. inside means nothing is lost, so convert. outside means something is lost, so refuse.
   3. no dtype is ruled out in advance: complex, uint64 and float128 are each handled by this same test.
   4. the target of a forced conversion is the smallest torch dtype whose set contains the source dtype's set. ply u2 and numpy uint16 both go to int32, ply u4 and numpy uint32 both go to int64.
4. The core design change in this task:
   1. the `PointCloud` class:
      1. new attr: meta data.
         1. what it is:
            1. meta data records whatever the original dtype is, upon construction. it records the source of the data, wherever the data comes from: a load from disk, a construction from a torch tensor or a numpy array, or addition or deletion of fields.
            2. it records the conceptual dtype, not the spelling of whichever system the field came from. a field entering as ply u2, as numpy uint16, or as an open3d UInt16 all record the same thing.
            3. a las bit-packed field is an ordinary unsigned integer. laspy materializes it as uint8, so uint8 is what enters the obj and uint8 is what the record stores. no special treatment.
         2. granularity: it is per-field, created when a field enters the obj and deleted when the field is removed.
         3. immutability: for each field, the record is never mutable. an overwrite of a field that already exists must NOT change the meta data.
      2. validation:
         1. `PointCloud` keeps validating xyz and rgb by field name.
         2. xyz is any floating point dtype.
         3. rgb is either
            1. an integer dtype, where the dtype's own range is the colour range, or
            2. a floating point dtype of any width holding 0 to 1.
            3. the two are told apart by dtype and never by inspecting the values, the same way `validate_vertex_color` tells mesh vertex colours apart.
            4. it differs from that mesh rule in admitting any integer width and any float width, because ply stores colours as u1 while las stores them as uint16, and a reader neither widens nor narrows what the file holds.
   2. consumers/users of `PointCloud`:
      1. any consumer of PointCloud in Pylon should be adjusted to work with the new design of PointCloud and its I/O.
      2. user of PointCloud obj may however modify the fields, but the meta data stays constant and immutable once created.
      3. the meta data travels with the field, so Select preserves it.
      4. point cloud I/O:
         1. no field name is special in I/O. xyz, rgb, indices, feat, colors, normals are ordinary fields.
         2. load point cloud
            1. preserves everything whenever possible, and converts dtype only for the mismatch between torch and the format it is reading. a reader never widens a field it builds: xyz records the dtype the file stores its coordinate columns in, so an f4 ply gives float32 xyz and an f8 ply gives float64 xyz.
            2. it takes an optional arg to override the dtype a field is loaded as, the same way save point cloud does.
         3. save point cloud
            1. strictly follows the meta data. it does not need to be aware of the dtype mismatch at all.
            2. meta data defaults to that version in the PointCloud obj, but overridable by an optional arg to control how it wants the field to be saved as.
            3. defensive programming: save point cloud casts each field to the recorded dtype and asserts the cast is lossless.
5. what becomes stale design:
   - the color rescale that guesses a [0, 1] range from the values and multiplies by 255
   - the narrowing of every integer field to i4
   - writing xyz as f4 whatever its dtype
   - the _seg filename test that casts feat to int64
   - the colors and pos aliases
   - PointCloud requiring indices to be int64. Select asserts it at the point of use instead

## 2. Definition of Done

1. Skeleton design and code conformance to skeleton both done. Tests in-scope all passes.
2. This branch is rebased onto latest `main`.
3. Confirmation message that this task is all done and this branch is good to merge.

### 2.1. Task Scope

1. load: .pth, .ply, .pcd, .las, .laz, .off, .txt. save: .ply. neither expands.
2. constructing a `PointCloud` from numpy arrays is in scope. the obj always stores torch tensors.
