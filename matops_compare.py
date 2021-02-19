from numpy.random import uniform
from tensorflow import constant
from timeit import timeit
import sys
# a (slow) baseline to compare to
def py_matmul(A,B):
    AB = [[0 for __ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                AB[i][j] += A[i][k] * B[k][j]
    return AB
from numpy import matmul as np_matmul
from tensorflow.linalg import matmul as tf_matmul

def main():
    n = int(sys.argv[1])
    t = int(sys.argv[2])
    for i in range(t):
        width=2**i
        shape = (width,width)
        A_np = uniform(-1,1,shape)
        B_np = uniform(-1,1,shape)
        A_py = list(A_np)
        B_py = list(B_np)
        A_tf = constant(A_np)
        B_tf = constant(B_np)
        print((n,t,width))
        print('py:', timeit(lambda: py_matmul(A_py,B_py),number=n)/n)
        print('np:', timeit(lambda: np_matmul(A_np,B_np),number=n)/n)
        print('tf:', timeit(lambda: tf_matmul(A_tf,B_tf),number=n)/n)
if __name__ == '__main__':
    main()
