goal: implement batched pc render

## 1. Guidelines

Batched rendering needs the tensors of every camera in the batch to share one size. So a new arg `valid` is added to the rasterizers, marking which points each camera kept.

### 1.1. Solution Constraints

the dtype and device of `Cameras` are
1. optional args default to None
2. resolved to those of `extrinsics` arg when None
3. `extrinsics` always follows the given `dtype` and/or `device` when they are not None.

i.e., the mental model is:
1. the dtype and device of the stored extrinsics matrix is a consequence of the given dtype and device, rather than the other way around.
2. the stored dtype and device are also consequences of the given dtype and device, rather than the other way around.
3. the only exception is when dtype or device is given None, in which case you resolve it to those of the given extrinsics matrix.

`transform_intrinsics` must NOT call any helper that serves convention conversion.

### 1.2. Explicitly and Strictly Banned Terms

- winner

## 2. Definition of Done

Empirically proved the following equivalence:
1. processing a single camera as a batch is equivalent as processing it using code on main.
2. processing a batched cameras is equivalent in results (not in speed) as processing one by one, using code on branch.
