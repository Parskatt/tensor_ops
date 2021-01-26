import numpy as np
from tensor_ops import Tensor

def test_ops():
    A = Tensor(np.random.randn(5,3,7))
    B = Tensor(np.random.randn(5,7,3))
    X = (A['abc']*B['acb'])['ac']
    Y = (A['abc']+B['acb'])['ac']**2
    Z = (A['abc']==B['acb'])['abc']
