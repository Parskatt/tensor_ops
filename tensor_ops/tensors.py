import numpy as np
from functools import reduce
from collections import OrderedDict

class TensorExpression:
    def __init__(self,expr_type,*tensors):
        self.tensors = tensors
        self.expr_type = expr_type
    def __getitem__(self,names):
        if self.expr_type == "mul":
            try:
                expr = ",".join([tensor.names for tensor in self.tensors])
            except:
                # Attempt naive multiplication if a tensor is missing names
                return Tensor(reduce(lambda x, y: x * y, [tensor._data for tensor in self.tensors]))
            expr += "->"
            expr += names
            return Tensor(np.einsum(expr,*[tensor._data for tensor in self.tensors]))
        elif self.expr_type == "add":
            try:
                all_names = "".join([tensor.names for tensor in self.tensors])
            except:
                # Attempt naive addition if a tensor is missing names
                return Tensor(reduce(lambda x, y: x + y, [tensor._data for tensor in self.tensors]))
            order,K,used_names = self.names_to_order([tensor.names for tensor in self.tensors])
            tensors = self.permute_to_order(self.tensors,order,K)
            out = Tensor(reduce(lambda x, y: x + y, [tensor._data for tensor in tensors]),used_names)[names]
            return out
        elif self.expr_type == "eq":
            try:
                all_names = "".join([tensor.names for tensor in self.tensors]) 
            except:
                # Attempt naive equality if a tensor is missing names
                return Tensor(reduce(lambda x, y: x == y, [tensor._data for tensor in self.tensors]))
            order,K,used_names = self.names_to_order([tensor.names for tensor in self.tensors])
            tensors = self.permute_to_order(self.tensors,order,K)
            out = Tensor(reduce(lambda x, y: x == y, [tensor._data for tensor in tensors]),used_names)[names]
            return out

    def names_to_order(self,names):
        k=0
        used = OrderedDict()
        order = []
        for tensor_names in names:
            tensor_order = []
            for c in tensor_names:
                if c in used:
                    tensor_order.append(used[c])
                else:
                    used[c] = k
                    tensor_order.append(k)
                    k += 1
            order.append(tensor_order)
        order = np.array(order)
        return order,k,"".join(used.keys())
    
    def permute_to_order(self,tensors,orders,k):
        new_tensors = []
        for tensor,order in zip(tensors,orders):
            fix_order = np.argsort(order)
            sorted_order = np.sort(order)
            tensor = tensor._permute(fix_order)
            dims = k*["None"]
            for o in order:
                dims[o] = ":"
            new_tensors.append(tensor[dims]) #Gimmie this https://stackoverflow.com/questions/55496700/starred-expression-inside-square-brackets
        return new_tensors

            
class Tensor:
    def __init__(self,data,names=None):
        self._data = np.array(data)
        self.names = names
        self.shape = self._data.shape

    def __getitem__(self,*index):
        if isinstance(index[0],str):
            if self.names is not None:
                return TensorExpression("mul",self)[index[0]]
            else:
                x = Tensor(self._data,*index)
                return x
        else:
            data = eval(f"self._data[{','.join([str(i) for i in index[0]])}]") # Gimmie this: https://stackoverflow.com/questions/55496700/starred-expression-inside-square-brackets
            return Tensor(data,self.names)

    def __mul__(self,other):
        if not isinstance(other,Tensor):
            self._data = other*self._data
            return self
        return TensorExpression("mul",self,other)
    
    def __add__(self,other):
        if not isinstance(other,Tensor):
            x = Tensor(self._data+other,self.names)
            return x
        return TensorExpression("add",self,other)
    
    def __pow__(self,gamma):
        return Tensor(self._data**gamma,self.names)

    def __eq__(self,other):
        if not isinstance(other,Tensor):
            x = Tensor(self._data==other,self.names)
            return x
        return TensorExpression("eq",self,other)
    
    def _permute(self,order):
        perm_names = "".join([self.names[o] for o in order])
        return Tensor(np.transpose(self._data,order),perm_names)
        
    def __repr__(self):
        return self._data.__repr__()