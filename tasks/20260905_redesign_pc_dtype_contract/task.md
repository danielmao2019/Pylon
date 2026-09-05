goal: re-design pc dtype contract/provenance

## Table of Contents <!-- omit in toc -->

- [1. Guidelines](#1-guidelines)
  - [1.1. Proposed Solution](#11-proposed-solution)
- [2. Definition of Done](#2-definition-of-done)
  - [2.1. Task Scope](#21-task-scope)

----------

## 1. Guidelines

### 1.1. Proposed Solution

1. The principle, in one sentence: load point cloud preserves everything whenever possible and only dtype converts for the dtype system mismatch between torch and the format it is reading.
2. the fundamental root cause is the dtype system mismatch: torch has no uint16 and no uint32, so ply u2 loads as int32 and ply u4 loads as int64. everything else loads unchanged: i1 as int8, u1 as uint8, i2 as int16, i4 as int32, f4 as float32, f8 as float64, b1 as bool.
3. The core design change in this task:
   1. add meta data in `PointCloud`:
      1. in addition to the fields being loaded, a meta data is created and stored inside the PointCloud obj returned from load.
      2. meta data records whatever the original dtype is, upon construction.
      3. it is per-field, created when a field enters the obj and deleted when the field is removed.
      4. it records the conceptual dtype, not the spelling of whichever system the field came from. a field entering as ply u2, as numpy uint16, or as an open3d UInt16 all record the same thing.
      5. a las bit-packed field is an ordinary unsigned integer. laspy materializes it as uint8, so uint8 is what enters the obj and uint8 is what the record stores. no special treatment.
   2. consumers/users of `PointCloud`:
      1. any consumer of PointCloud in Pylon should be adjusted to work with the new design of PointCloud and its I/O.
      2. user of PointCloud obj may however modify the fields, but the meta data stays constant and immutable once created.
      3. the meta data travels with the field, so Select preserves it.
      4. point cloud I/O:
         1. no field name is special in I/O. xyz, rgb, indices, feat, colors, normals are ordinary fields.
         2. save point cloud
            1. strictly follows the meta data. it does not need to be aware of the dtype mismatch at all.
            2. meta data defaults to that version in the PointCloud obj, but overridable by an optional arg to control how it wants the field to be saved as.
            3. defensive programming:
               1. save point cloud casts each field to the recorded dtype and asserts the cast is lossless.
               2. torch int64 has no ply column, so it is refused unless overridden.
4. what becomes stale design:
   - the colour rescale that guesses a [0, 1] range from the values and multiplies by 255
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
