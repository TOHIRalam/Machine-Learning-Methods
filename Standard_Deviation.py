from cv2 import sqrt
from numpy import mean
import statistics
import numpy

# Standard deviation is a number that describes how spread out the values are.
# A low standard deviation means that most of the numbers are close to the mean (average) value.
# A high standard deviation means that the values are spread out over a wider range.
# So, it measures the dispersion of dataset relative to its mean and calculated as the square root of the variance. 
# The formula is "It is the square root of the variance"

# Variance: How far each number in the set is from the mean. It is a measurement of spread between numbers in a dataset.
# It is the average of the squared differences from the mean. 

# Standard Deviation with Example  
# --------------------------------

# Let's measure dog's height in milimeatres 
dogs_height = [600, 470, 170, 430, 300]

# And, number of datapoints we have-
number_of_datapoints = len(dogs_height)

# 1. Find the average or mean value of the data points ( avg )
# So, firstly lets get the average value of the datapoints 
Sum = 0
for i in dogs_height:
    Sum += i 
average = Sum/number_of_datapoints
print(f"Average: {average}")

# Find average by using mean function 
# average = mean(dogs_height)

# Find average by using sum() len() built in funcitons
# average = sum(dogs_height)/len(dogs_height)

# 2. Find the absolute difference of datapoints A[i] from average ( avg )
#    This means subtract the mean from the datapoints each time. 
absolute_differences = []
for i in dogs_height:
    absolute_differences.append(abs(i - average))
print(f"Absolute Differences: {absolute_differences}")

# 3. Square the differences of the data points A[i] from average avg Abs(A[i] - avg)^2
squared_differences = []
for i in absolute_differences:
    squared_differences.append(i**2)
print(f"Squared Differences: {squared_differences}")

# 4. Divide the sum of the square by number of datapoints SUM(Abs(A[i] - avg)^2)/n to get the variance 
variance = sum(squared_differences)/number_of_datapoints
print(f"Variance: {variance}")

# 5. Now, square root the variance to get the standard deviation 
standard_deviation = sqrt(variance)
print(f"Standard Deviation: {standard_deviation}")

################################################
# We can also get the standard deviation by using std() function from numpy module 
standard_deviation2 = numpy.std(dogs_height)
print(f"Standard Deviation: {standard_deviation2}")
