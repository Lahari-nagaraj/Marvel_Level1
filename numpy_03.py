import numpy as np  # Step 1: Import NumPy

# Step 2: Create a small array
small_array = np.array([[1, 2], [3, 4]])

# Step 3: Repeat the small array (2 times along rows, 3 times along columns)
large_array = np.tile(small_array, (2, 3))

# Step 4: Print the result
print("Repeated Array:")
print(large_array)

# Step 1: Generate numbers from 1 to 9 in ascending order
ascending_array = np.arange(1, 10).reshape(3, 3)

# Step 2: Print the array
print("\nArray with Elements in Ascending Order:")
print(ascending_array)
