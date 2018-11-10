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

# creates an array of the given shape

x = np.random.rand(10)
x

array([ 0.23682728,  0.62756037,  0.94301737,  0.69327749,  0.83978074,
        0.76425897,  0.77574374,  0.91636499,  0.78126268,  0.81175193])
        
     

# Numpy.random sample
Generates a random sample given from a 1-d array parameters = if an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)

Output shape. If the given shape is for example (m, n, k), then m n k samples are takenm the default will be none,and a single value will be returned.

If the sample is with or without replacement

p : 1-D array-like, optional then i The probabilities associated with each entry in A. If we are not given the sample then it presumes that a uniform distribution over all otehr inpupts.

Returns = sample = single item ndarray, generated random samples

Raises: valueError = if A is an integer less than 0, or if A or P are not 1 dimensional if a is an array-like of size 0, if p is not a vector of probabilities, if a and p have different lengths, or if replace=False and the sample size is greater than the population size


# Numpy.random choice examples

import numpy as np
>>> a = [1,4,1,3,3,2,1,4]
>>> np.random.choice(a)
>>> 4
>>> a
>>> [1,4,1,3,3,2,1,4]


or

import random
from scipy import *
print(random)

#this is the same as np.radom.randint.
np.random.choice(5, 3)
array([0, 3, 4])

out = array([0, 3, 4])

np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
array([3, 3, 0])

out = array([3, 3, 0])

#This is the same as np.random.permutation
np.random.choice(5, 3, replace=False)
array([3,1,0])

array([3, 1, 0])

np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])

out = array([2, 3, 0])















# Numpy.random.beta


Numpy.random.beta is a special case of the Dirichlet distribution, and is also realted to the Gamma distribution. it includes the probality distibtution function, where the normalisation B is the beta function. These are the parameters = 1,float or array like of floats (alpha/non negative)2, float or array like of floats (beta/non negative) 3. size = int or tuple of ints, optional (output shape if given) Eg: e.g., (m, n, k), then m n k samples are drawn. If size is None (default), a single value is returned if a and b are both scalars. Otherwise, np.broadcast(a, b).size samples are drawn.

Returns: our = ndarray or scalar (samples drawn from teh parameterized beta distribution.

# Numpy.random.binomial

This draws samples from a binomial distribution with specified parameters, n trials and p probablility of success where n an integere >=0 and p is in the interval between 0-1, it may be inputted as a flaot but in turn will be changed into an integer. Below are the parameter and returns Parameters:

n : int or array_like of ints

p : float or array_like of floats

This is the Parameter of the distribution, >= 0 and <=1.

size : int or tuple of ints, optional

Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if n and p are both scalars. Otherwise, np.broadcast(n, p).size samples are drawn.

Returns:

out : ndarray or scalar

#n is the number of trial and p is the probability of success and then n is the number of successed. if we were trying to estimate the standard error of the proportion of the population bu using the random sample then the normal distribution works well under the prduct pn and x<5, where p is the population proportion estimate. where n is the no of samples in which case the binomial distribution is used instead. example is 4 people with autism and 11 without autism, then p = 4/15 = 27%, then 0.2715 = 4. showing the binomial distribution should be used in this case.






##References

https://www.google.com/search?q=expalin+simple+random+data&ie=utf-8&oe=utf-8&client=firefox-b
https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Descriptive_Statistics-Summary_Tables.pdf
https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.binomial.html
https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781785884870/2/ch02lvl1sec20/python-random-numbers-in-jupyter

https://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/
https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html#numpy.random.randn


https://www.ohbmbrainmappingblog.com/blog/a-brief-overview-of-permutation-testing-with-examples
https://matthew-brett.github.io/les-pilot/permuter.html






