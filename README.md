# (WIP!) Better Tensor Ops

Proof of concept of a more intuitive way of doing tensor operations in python.


This is a high level library, i.e. only the front api of numpy/pytorch etc is used.
The Tensor class acts as wrapper around the array, delegating operations to the built in versions of the corresponding framework.
This means that certain operations may be inefficient, since they might rely on broadcasting etc.
Especially regular indexing is extremely inefficient as it relies on evaluating strings as a hack to enable the same syntax to be used.

Inspired by https://github.com/Jutho/TensorOperations.jl

## Examples

```python
import numpy as np
from tensor_ops import Tensor

A = Tensor(np.random.randn(5,10,6,7))
B = Tensor(np.random.randn(5,7))
C = Tensor(np.random.randn(5,8))

D = (A['abcd']+B['ad'])['abcd']
E = (A['abcd']*B['ad'])['da']
F = (A['abcd']==C['aj'])['ja']

```
Tensors can also be multiplied and added without providing names as long as they are of numpy compatible broadcastable shapes. 
However, Tensors will not behave as the underlying array when used in e.g. numpy functions.
