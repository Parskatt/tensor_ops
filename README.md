# Better Tensor Ops

Proof of concept of a more intuitive way of doing tensor operations in python.

This is a high level library, i.e. only the front api of numpy/pytorch etc is used.
This means that certain operations may be inefficient, since they might rely on broadcasting etc.

Inspired by https://github.com/Jutho/TensorOperations.jl