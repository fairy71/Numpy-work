import numpy as np

def find_missing_number(arr):
    """
    Finds the missing number in a NumPy array of size n-1, 
    containing numbers from 1 to n.
    
    Args:
        arr (np.array): A NumPy array with one number missing.

    Returns:
        int: The missing number.
    """
    n = len(arr) + 1
    
    # Calculate the expected sum of numbers from 1 to n
    expected_sum = n * (n + 1) // 2
    
    # Calculate the actual sum of the numbers in the array
    actual_sum = np.sum(arr)
    
    # The difference is the missing number
    missing_number = expected_sum - actual_sum
    
    return missing_number

# --- Example usage ---
# Create an array where the number 7 is missing
original_list = [1, 2, 3, 4, 5, 6, 8, 9, 10]
numpy_array = np.array(original_list)

print(f"Original array: {numpy_array}")

# Find the missing number
missing_num = find_missing_number(numpy_array)

print(f"The missing number is: {missing_num}")

# Another example with a different missing number
original_list_2 = [3, 1, 2, 5, 6]
numpy_array_2 = np.array(original_list_2)

print(f"\nOriginal array: {numpy_array_2}")

# Find the missing number
missing_num_2 = find_missing_number(numpy_array_2)

print(f"The missing number is: {missing_num_2}")
