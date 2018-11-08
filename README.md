# My-investigaton-into-numpyNumPy.Random package

The overall purpose of the NumPy. Random package is for scientific computing with python. It has a powerful N-dimensional array object, some sophisticated functions, tools for integrating C/C++ and Fortran code. It’s also used for linear algebra, Fourier transform and random number capabilities.
NumPy can also be used as an efficient multidimensional container of generic data, this means that arbitrary data types can be defined allowing NumPy to be faster and more efficient at working with other databases.
Numpy.random
Numpy.random.rand
This function gives random values in a given shape, for example you can have ? amount of rows, columns or overall depth.
Eg: 

# import pandas.
import pandas as pd

# load the Anscombe data set from a URL

df = pd.read_csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/anscombe.csv")


# import numpy
import numpy as np

# creates an array of the given shape

x = np.random.rand(10)
x
array([ 0.80218759,  0.93129738,  0.86817952,  0.12530837,  0.46180936,
        0.51364694,  0.63378701,  0.52573197,  0.74699591,  0.4194375 ])

# import matplotlib
import matplotlib.pyplot as plt
plt.hist(x)
plt.show()



x = np.random.uniform(0,10,100)



plt.hist(x)
plt.show()





numpy.random.rand(d0, d1, ..., dn)
Random values in a given shape.
Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).


