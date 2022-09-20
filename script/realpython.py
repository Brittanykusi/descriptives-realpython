import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

# descriptive stats
x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x
x_with_nan

math.isnan(np.nan), np.isnan(math.nan)

y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y

y_with_nan

z_with_nan

# measures of central tendency #
# mean
mean_ = sum(x) / len(x)
mean_

mean_ = statistics.mean(x_with_nan)
mean_

#You can implement the weighted mean in pure Python by combining sum() with either range() or zip():
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

#You can also calculate this measure with statistics.harmonic_mean():
hmean = statistics.harmonic_mean(x)
hmean

#If you provide at least one negative number, then you’ll get statistics.StatisticsError:
statistics.harmonic_mean(x_with_nan)
statistics.harmonic_mean([1, 0, 2])
statistics.harmonic_mean([1, 2, -2])  # Raises StatisticsError

#A third way to calculate the harmonic mean is to use scipy.stats.hmean():
scipy.stats.hmean(y)
scipy.stats.hmean(z)

#The geometric mean is the 𝑛-th root of the product of all 𝑛 elements 𝑥ᵢ 
# in a dataset 𝑥: ⁿ√(Πᵢ𝑥ᵢ), where 𝑖 = 1, 2, …, 𝑛. The following figure illustrates 
# the arithmetic, harmonic, and geometric means of a dataset:

gmean = 1
for item in x:
    gmean *= item
gmean **= 1 / len(x)
gmean

#median
#The sample median is the middle element of a sorted dataset. 
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])

median_

median_ = statistics.median(x[:-1])
median_

#median_low() and median_high() are two more functions related to the median in the Python statistics library. They always return an element from the dataset:

#If the number of elements is odd, then there’s a single middle value, 
# so these functions behave just like median().

#If the number of elements is even, then there are two middle values. 
# In this case, median_low() returns the lower and median_high() the higher middle value.
statistics.median_low(x[:-1])

statistics.median_high(x[:-1])

#Unlike most other functions from the Python statistics library, median(), median_low(),
# and median_high() don’t return nan when there are nan values among the data points:
statistics.median(x_with_nan)
statistics.median_low(x_with_nan)
statistics.median_high(x_with_nan)

#mode
#The sample mode is the value in the dataset that occurs most frequently. 
# If there isn’t a single such value, then the set is multimodal since it has 
# multiple modal values.
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

#You can obtain the mode with statistics.mode() and statistics.multimode():
mode_ = statistics.mode(u)
mode_
mode_ = statistics.multimode(u)
mode_

#Pandas Series objects have the method .mode() that handles multimodal values well 
# and ignores nan values by default:
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()


v.mode()


w.mode()

# measures of variability #
#Here’s how you can calculate the sample variance with pure Python:
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_

#This approach is sufficient and calculates the sample variance well.
var_ = statistics.variance(x)
var_

#If you have nan values among your data, then statistics.variance() will return nan:
statistics.variance(x_with_nan)

#You can also calculate the sample variance with NumPy. 
# You should use the function np.var() or the corresponding method .var():
var_ = np.var(y, ddof=1)
var_

var_ = y.var(ddof=1)
var_

# This is consistent with np.mean() and np.average(). 
# If you want to skip nan values, then you should use np.nanvar():
np.nanvar(y_with_nan, ddof=1)

#standard deviation
std_ = var_ ** 0.5
std_

#You can get the standard deviation with NumPy in almost the same way. 
# You can use the function std() and the corresponding method .std() to calculate 
# the standard deviation. If there are nan values in the dataset, then they’ll return nan.
# To ignore nan values, you should use np.nanstd(). You use std(), .std(), and nanstd() from 
# NumPy as you would use var(), .var(), and nanvar():


np.std(y, ddof=1)

y.std(ddof=1)

np.std(y_with_nan, ddof=1)

y_with_nan.std(ddof=1)

np.nanstd(y_with_nan, ddof=1)

#skewness
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
         * n / ((n - 1) * (n - 2) * std_**3))
skew_

y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)
scipy.stats.skew(y_with_nan, bias=False)

# percentiles
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)
statistics.quantiles(x, n=4, method='inclusive')

y = np.array(x)
np.percentile(y, 5)

np.percentile(y, 95)

#The percentile can be a number between 0 and 100 like in the example above, 
# but it can also be a sequence of numbers:
np.percentile(y, [25, 50, 75])
array([ 0.1,  8. , 21. ])
np.median(y)

#If you want to ignore nan values, then use np.nanpercentile() instead:
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan

np.nanpercentile(y_with_nan, [25, 50, 75])

#NumPy also offers you very similar functionality in quantile() and nanquantile(). 
# If you use them, then you’ll need to provide the quantile values as the numbers between 
# 0 and 1 instead of percentiles:
np.quantile(y, 0.05)

np.quantile(y, 0.95)

np.quantile(y, [0.25, 0.5, 0.75])

np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])

#range
#The range of data is the difference between the maximum and minimum element 
# in the dataset. You can get it with the function np.ptp():
np.ptp(y)

np.ptp(z)

np.ptp(y_with_nan)

np.ptp(z_with_nan)

#Alternatively, you can use built-in Python, NumPy, or Pandas functions and methods
# to calculate the maxima and minima of sequences:
np.amax(y) - np.amin(y)

np.nanmax(y_with_nan) - np.nanmin(y_with_nan)

y.max() - y.min()

z.max() - z.min()

z_with_nan.max() - z_with_nan.min()

#summary of descriptive statistics
result = scipy.stats.describe(y, ddof=1, bias=False)
result


#describe() returns an object that holds the following descriptive statistics:

#nobs: the number of observations or elements in your dataset
#minmax: the tuple with the minimum and maximum values of your dataset
#mean: the mean of your dataset
#variance: the variance of your dataset
#skewness: the skewness of your dataset
#kurtosis: the kurtosis of your dataset
#result.nobs

result.minmax[0]  # Min

result.minmax[1]  # Max

result.mean

result.variance

result.skewness

result.kurtosis

#Pandas has similar, if not better, functionality. Series objects have the method .describe():
result = z.describe()
result

#It returns a new Series that holds the following:

#count: the number of elements in your dataset
#mean: the mean of your dataset
#std: the standard deviation of your dataset
#min and max: the minimum and maximum values of your dataset
#25%, 50%, and 75%: the quartiles of your dataset
result['mean']

result['std']

result['min']

result['max']

result['25%']

result['50%']

result['75%']

#Measures of Correlation Between Pairs of Data

x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

#covariance
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy

#NumPy has the function cov() that returns the covariance matrix:
cov_matrix = np.cov(x_, y_)
cov_matrix

#The other two elements of the covariance matrix are equal and 
# represent the actual covariance between x and y:
cov_xy = cov_matrix[0, 1]
cov_xy

cov_xy = cov_matrix[1, 0]
cov_xy

#Correlation Coefficient
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

#scipy.stats has the routine pearsonr() that calculates the 
#correlation coefficient and the 𝑝-value:
r, p = scipy.stats.pearsonr(x_, y_)
r

p

#Similar to the case of the covariance matrix, you can apply np.corrcoef() 
#with x_ and y_ as the arguments and get the correlation coefficient matrix:
corr_matrix = np.corrcoef(x_, y_)
corr_matrix

#You can get the correlation coefficient with scipy.stats.linregress():
scipy.stats.linregress(x_, y_)

#working with 2d data
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a
#Now you have a 2D dataset, which you’ll use in this section. 
# You can apply Python statistics functions and methods to it just as you would to 1D data:
np.mean(a)

a.mean()

np.median(a)

a.var(ddof=1)

#Let’s see axis=0 in action with np.mean():
np.mean(a, axis=0)

a.mean(axis=0)

#If you provide axis=1 to mean(), 
# then you’ll get the results for each row:
np.mean(a, axis=1)

a.mean(axis=1)

#The parameter axis works the same way 
# with other NumPy functions and methods:
np.median(a, axis=0)

np.median(a, axis=1)

a.var(axis=0, ddof=1)

a.var(axis=1, ddof=1)

#This is very similar when you work with SciPy statistics functions. But remember that in this case, 
# the default value for axis is 0:
scipy.stats.gmean(a)  # Default: axis=0

scipy.stats.gmean(a, axis=0)

#If you specify axis=1, then you’ll get the calculations across all columns, that is for each row:
scipy.stats.gmean(a, axis=1)

#You can get a Python statistics summary with a single function call for 2D data with scipy.stats.describe(). 
# It works similar to 1D arrays, but you have to be careful with the parameter axis:
scipy.stats.describe(a, axis=None, ddof=1, bias=False)

scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0

scipy.stats.describe(a, axis=1, ddof=1, bias=False)

#You can get a particular value from the summary with dot notation:
result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean

#dataframes
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df

df.mean()
df.var()

#If you want the results for each row, then just specify the parameter axis=1:
df.mean(axis=1)
df.var(axis=1)
df['A']

#Now, you have the column 'A' in the form of a Series object and you can apply the appropriate methods:
df['A'].mean()

df['A'].var()

#visualizing data
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#boxplots
#The box plot is an excellent tool to visually represent descriptive statistics of a given dataset.
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

#histogram
hist, bin_edges = np.histogram(x, bins=10)
hist

bin_edges

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

#pie charts
x, y, z = 128, 256, 1024
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

#bar charts
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)

fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#x-y-plots
x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

#heatmaps
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()

# You can obtain the heatmap for the correlation coefficient matrix following the same logic:
matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()