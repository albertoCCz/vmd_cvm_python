import numpy as np 
cimport numpy as np 

# Define complex random array
real = np.random.random(7)
img  = np.random.random(7) + 1j
arr  = real + img
cdef np.complex_t[:] arrc = arr

real2 = np.random.random(7)
img2  = np.random.random(7) + 1j
arr2  = real + img
cdef np.complex_t[:] arr2c = arr2

# cdef np.complex_t[:] temp = np.empty(len(arr), dtype=np.complex)
# cdef size_t i
# for i in range(len(arr)):
#     temp[i] = arrc[i] + arr2c[i]

# Can we sum them as memoryviews?
print(f"Python sum is: {arr + arr2}")
print(f"Cython sum is: {(np.real(arrc) + np.real(arr2c)) + (np.imag(arrc) + np.imag(arr2c)) * 1j}")
