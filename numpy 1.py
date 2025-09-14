import numpy as np

def find_missing_number(arr):
    
    n = len(arr) + 1
    
    expected_sum = n * (n + 1) // 2
    
    actual_sum = np.sum(arr)
    
    missing_number = expected_sum - actual_sum
    
    return missing_number





original_list = [1, 2, 3, 4, 5, 6, 8, 9, 10]
numpy_array = np.array(original_list)

print(f"Original array: {numpy_array}")

missing_num = find_missing_number(numpy_array)

print(f"The missing number is: {missing_num}")

original_list_2 = [3, 1, 2, 5, 6]
numpy_array_2 = np.array(original_list_2)

print(f"\nOriginal array: {numpy_array_2}")

missing_num_2 = find_missing_number(numpy_array_2)

print(f"The missing number is: {missing_num_2}")
