#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#NUMPY ASSIGNMENT

1. Write a NumPy program to create a structured array from given student
name, height, class and their data types. Now sort the array on height
2. Write a NumPy program to create a structured array from given student
name, height, class and their data types. Now sort by class, then
height if class are equa
3. Replace NumPy array elements that doesn’t satisfy the given condition
4. Return the indices of elements where the given condition is satisfied
5. Replace NaN values with average of columns
6. Replace negative value with zero in numpy array
get_ipython().run_line_magic('pinfo', 'positions')
8. Find indices of elements equal to zero in a NumPy array
9. How to Remove columns in Numpy array that contains non-numeric
get_ipython().run_line_magic('pinfo', 'values')
get_ipython().run_line_magic('pinfo', 'array')
11. Get row numbers of NumPy array having element larger than X
12. Get filled the diagonals of NumPy array
13. Check elements present in the NumPy array


# In[4]:


#1. Write a NumPy program to create a structured array from given student
# name, height, class and their data types. Now sort the array on height

import numpy as np
data_type = [('name', 'S15'), ('class', int), ('height', float)]
students_details = [('pooja', 5, 48.5), ('shabnam', 6, 52.5),('manoj', 5, 42.10), ('sahil', 5, 40.11)]
# create a structured array
students = np.array(students_details, dtype=data_type)   
print("Original array:")
print(students)
print("Sort the array on height :")
print(np.sort(students, order=['height']))






# In[5]:


# Write a NumPy program to create a structured array from given student name ,height, class and their
# data types  Now sort by class, then height if class are equal

import numpy as np
data_type = [('name', 'S15'), ('class', int), ('height', float)]
students_details = [('pooja', 5, 48.5), ('manoj', 6, 52.5),('sahil', 5, 42.10), ('prachi', 5, 40.11)]
# create a structured array
students = np.array(students_details, dtype=data_type)   
print("Original array:")
print(students)
print("Sort by class, then height if class are equal:")
print(np.sort(students, order=['class', 'height']))



# In[7]:


# Replace NumPy array elements that doesn’t satisfy the given condition


import numpy as np
  
# Creating a 1-D Numpy array
n_arr = np.array([75.42436315, 42.48558583, 60.32924763])
print("Given array:")
print(n_arr)
  
print("\nReplace all elements of array which are greater than 50. to 15.50")
n_arr[n_arr > 50.] = 15.5123498
  
print("New array :\n")
print(n_arr)



# In[32]:


# Return the indices of elements where the given condition is satisfied

import numpy as np
arr = np.array([2, 5, 6, 13, 10, 24, 34, 90, 11, 67])
result = np.where(arr < 15) 
print("original array: ",arr)
print('New array:',result)


# In[22]:


##Replace NaN values with average of columns
import numpy as np
import pandas as pd
# A dictionary with list as values
sample_dict = { 's1': [10, 20, np.NaN, np.NaN],
's2': [5, np.NaN, np.NaN, 29],
's3': [15, np.NaN, np.NaN, 11],
's4': [21, 22, 23, 25],
'Subjects': ['Maths', 'Finance', 'History', 'Geography']}
# Create a DataFrame from dictionary
df = pd.DataFrame(sample_dict)
df = df.set_index('Subjects')

print(df)




# In[23]:


# Replace negative value with zero in numpy array
import numpy as np
  
ini_array1 = np.array([1, 2, -3, 4, -5, -6])
  

print("initial array", ini_array1)
ini_array1[ini_array1<0] = 0
  
# printing result
print("New resulting array: ", ini_array1)


# In[26]:


#How to get values of an NumPy array at certain index positions?
a = [0,88,26,3,48,85,65,16,97,83,91]
import numpy as np
arr = np.array(a)
ind_pos = [1,5,7]
arr[ind_pos]


# In[29]:


# Find indices of elements equal to zero in a NumPy array
import numpy as np
nums = np.array([1,0,2,0,3,0,4,5,6,7,8])
print("Original array:")
print(nums)
print("Indices of elements equal to zero of the said array:")
result = np.where(nums == 0)[0]
print(result)


# In[35]:


#How to Remove columns in Numpy array that contains non-numeric values
import numpy as np
x = np.array([[1,2,3], [4,5,np.nan], [7,8,9], [True, False, True]])
print("Original array:")
print(x)
print("Remove all non-numeric elements of the said array")
print(x[~np.isnan(x).any(axis=1)])



# In[37]:


#How to access different rows of a multidimensional NumPy array?
import numpy as np
 
# Creating a 3X3 2-D Numpy array
arr = np.array([[10, 20, 30], 
               [40, 5, 66], 
               [70, 88, 94]])
 
print("Given Array :")
print(arr)
 
# Access the First and Last rows of array
res_arr = arr[[0,1]]
print("\nAccessed Rows :")
print(res_arr)


# In[38]:


#Get row numbers of NumPy array having element larger than X
import numpy
  
# create numpy array
arr = numpy.array([[1, 2, 3, 4, 5],
                  [10, -3, 30, 4, 5],
                  [3, 2, 5, -4, 5],
                  [9, 7, 3, 6, 5] 
                 ])
X = 6
print("Given Array:\n", arr)
  
output  = numpy.where(numpy.any(arr > X,
                                axis = 1))
print("Result:\n", output)


# In[39]:


# Get filled the diagonals of NumPy array
import numpy as np
  
array = np.array([[1, 2], [2, 1]])
np.fill_diagonal(array, 5)
  
print(array)


# In[40]:


#Check elements present in the NumPy array

import numpy as np

array = np.array([4,6,0,0,0,4,89])
print(x)
print(np.any(array))


# In[ ]:




