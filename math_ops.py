from random import random
import sys
from timeit import timeit
def main():
    n = int(sys.argv[1])
    t = int(sys.argv[2])
    for i in range(t):
        width=2**i
        shape = (width,width)
        print((n,i,width))
        print('+:', timeit(lambda: width*random() + width*random(),number=n)/n)
        print('*:', timeit(lambda: width*random() * width*random(),number=n)/n)
        print('**:', timeit(lambda: width*random() ** width*random(),number=n)/n)
if __name__ == '__main__':
    main()
