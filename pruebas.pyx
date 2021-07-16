import numpy as np 
cimport numpy as np 

# Define ndarray
arr = np.random.random(7)

# Equivalent view
cdef np.float64_t[:] arrc = arr 

# Go and try to sort them
print(f"Sorted arr: {np.sort(arr)}")
print(f"Sorted arrc: {np.sort(arrc)}")
