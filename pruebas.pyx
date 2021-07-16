import numpy as np 
cimport numpy as np 

# Define complex random array
arr = np.random.random(7)
cdef np.float64_t[:] arrc = arr

# Can we flip memoryviews?
print(f"Python flip is: {np.flip(arr[:3])}")
print(f"Cython flip is: {list(np.flip(arrc[:3]))}")
