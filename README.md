## My-investigaton-into-numpyNumPy.Random package

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

# Simple random data

Simple Random data is similar to a random sample, as it is randomly selected, this method avoids bias and arguments or a chance that an individuals personal opinions or views could distort the datas reuslts. The word random in this scenario is not the exact same as the dictionary meaning of the word random, as it is not chosen without some sort of concious intention. there is a reason behind using this method. It is quiet complex even tho in theory it seems fairly simple. So down to a technical explanation of random data, or random sampling, is a set of n objects in a population of n objects where all possible samples are equally likely to happen.
this is an example i came across.
Put 100 balls into a bowl(this is now the population n)
If you imagine being asked then to select 10 balls from a bowl without looking (this is now our sample n)
if you were to look this could bias the sample!!
This is known as the lottery bowl method and works well with a small sample sie but in reality we would be expected to be working with a much larger sample population.

This method gives everyone a fair and equal chance to be in the selected sample. But bias is its downfall as it is very easy for it to creep in and effect the sample population. Now we will look at an example of how to perform simple random sampling. 



# Numpy.random.rand
numpy.random.rand(d0, d1, ..., dn)
Random values in a given shape.
Create an array of the given shape and populate it with r
nandom samples from a uniform distribution over [0, 1).









# Numpy.random.sample






# Numpy.random.beta




plt.hist(x)
plt.show()











