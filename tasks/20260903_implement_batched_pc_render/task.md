goal: implement batched pc render

## 1. Guidelines

Batched rendering needs the tensors of every camera in the batch to share one size. So a new arg `valid` is added to the rasterizers, marking which points each camera kept.

### 1.1. Explicitly and Strictly Banned Terms:

- winner

## 2. Definition of Done

Empirically proved the following equivalence:
1. processing a single camera as a batch is equivalent as processing it using code on main.
2. processing a batched cameras is equivalent in results (not in speed) as processing one by one, using code on branch.
