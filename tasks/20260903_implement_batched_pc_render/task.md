goal: implement batched pc render

Batched rendering needs the tensors of every camera in the batch to share one size. So a new arg `valid` is added to the rasterizers, marking which points each camera kept.
