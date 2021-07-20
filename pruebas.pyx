import numpy as np 
cimport numpy as np 

# Define complex random array
randomr = np.random.random(7)
randomi = np.random.random(7)
cdef np.complex_t[:] randomc = np.array(randomr + randomi * 1j, dtype=np.complex)

# Print memoryview
print(list(randomc))
print(randomc[1].real)

# Modify real part of values
cdef size_t i 
for i in range(randomc.shape[0]):
    randomc[i] = {'real':randomc[i].real + 1, 'imag':randomc[i].imag}

# Print modified memoryview
print(list(randomc))


# Can we flip memoryviews?
# print(f"Python flip is: {np.flip(arr)}")
# print(f"Cython flip is: {list(np.flip(arrc))}")
