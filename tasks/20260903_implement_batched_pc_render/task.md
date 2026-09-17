goal: implement batched pc render

## 1. Guidelines

Batched rendering needs the tensors of every camera in the batch to share one size. So a new arg `valid` is added to the rasterizers, marking which points each camera kept.

### 1.1. Solution Constraints

#### 1.1.1. Single Source of Truth of dtype and device

the dtype and device of `Cameras` are
1. optional args default to None
2. resolved to the dtype and device the given `intrinsics` and `extrinsics` share when None
3. `intrinsics` and `extrinsics` both always follow the given `dtype` and/or `device` when they are not None.

i.e., the mental model is:
1. the dtype and device of the stored intrinsics params and extrinsics matrix are consequences of the given dtype and device, rather than the other way around.
2. the stored dtype and device are also consequences of the given dtype and device, rather than the other way around.
3. the only exception is when dtype or device is given None, in which case you resolve it to the one the two given components share, neither component taking precedence over the other.
4. intrinsics and extrinsics of the same camera/cameras are two parts of the same object, so must always be on same device.

#### 1.1.2. Structure of Camera Intrinsics Transforms and Scaling

`transform_intrinsics` must NOT call any helper that serves convention conversion.

#### 1.1.3. Hierarchical Structure between Camera/Cameras and Intrinsics/Extrinsics

The validation must be hierarchically implemented. Any validation of camera/cameras must rely on those of intrinsics/extrinsics, if the logic does belong to intrinsics/extrinsics. Camera/cameras validators should work on the validations at their own hierarchical levels, and must never do any job of intrinsics/extrinsics.

### 1.2. Explicitly and Strictly Banned Terms

- winner

## 2. Definition of Done

Empirically proved the following equivalence:
1. processing a single camera as a batch is equivalent as processing it using code on main.
2. processing a batched cameras is equivalent in results (not in speed) as processing one by one, using code on branch.
