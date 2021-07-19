import numpy as np 
cimport numpy as np 

# Define complex random array
randoms = np.random.random(7)
cdef np.float64_t[:] crandoms = randoms
arr = np.empty(shape=len(randoms), dtype=np.float64)
cdef np.float64_t[:] arrc = np.empty(shape=len(randoms), dtype=np.float64)

# Assign values
cdef np.float64_t[:] crandoms_flipped = np.flip(crandoms)
arrc[:len(randoms)//2] = crandoms_flipped[:len(randoms)//2]

print(list(arrc))


# Can we flip memoryviews?
# print(f"Python flip is: {np.flip(arr)}")
# print(f"Cython flip is: {list(np.flip(arrc))}")
