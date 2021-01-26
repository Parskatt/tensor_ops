# Better Tensor Ops

Proof of concept of a more intuitive way of doing tensor operations in python.


This is a high level library, i.e. only the front api of numpy/pytorch etc is used.
The Tensor class acts as wrapper around the array, delegating operations to the built in versions of the corresponding framework.
This means that certain operations may be inefficient, since they might rely on broadcasting etc.
Especially regular indexing is extremely inefficient as it relies on evaluating strings as a hack to enable the same syntax to be used.


Inspired by https://github.com/Jutho/TensorOperations.jl
