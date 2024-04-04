import numpy as np

from function import *

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
c = C(b)
c.grad = np.array(1.0)
c.backward()
print(x.grad)
