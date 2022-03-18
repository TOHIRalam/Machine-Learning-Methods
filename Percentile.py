import numpy

# Percentile: the value below which a percentage of data falls.
# It is a number that describes the value that a given percent of the values are lower than.

# Let's say we have an array of the ages of all the people that lives in a street.
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

# What is the 75. percentile? The answer is 43, meaning that 75% of the people are 43 or younger.

##################### RULES ####################

# 1. Sort the values of the dataset 
# 2. Divide the percentile value by 100 
# 3. Multiply with the total number of dataset 
# 4. The result is the index value of the kth percentile 

ages.sort()

percent = 75
number_of_items = len(ages)

result = int((percent / 100) * (number_of_items)) 

print(f"Percentile: {ages[result]}")


# There is another way to do this in python using NumPy percentile method 

print(f"Percentile (numpy): {numpy.percentile(ages, percent)}")