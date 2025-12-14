import random
import csv

# Generate random digits
random_array = [random.randint(0, 9) for _ in range(10000)]
setup_array = [1, 4, 2, 7, 9, 0, 3, 4]

num_array = []
binary_array = []

for i in range(10000):

    # Shift left and append new random digit
    setup_array = setup_array[1:] + [random_array[i]]

    # Build 8-digit string
    num_str = "".join(str(d) for d in setup_array)

    # Convert to int ONLY for divisibility test
    num_val = int(num_str)

    num_array.append(num_str)  
    binary_array.append(1 if num_val % 32 == 0 else 0)

    print(f"Number: {num_str} Binary: {binary_array[-1]}")

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

split_index = int(0.8 * len(num_array))  # 80% train, 20% test

train_data = list(zip(num_array[:split_index], binary_array[:split_index]))
test_data  = list(zip(num_array[split_index:], binary_array[split_index:]))

# -----------------------------
# WRITE TRAIN FILE
# -----------------------------
with open("train_32.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["data", "label"])
    for num, label in train_data:
        writer.writerow([num, label])

# -----------------------------
# WRITE TEST FILE
# -----------------------------
with open("test_32.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["data", "label"])
    for num, label in test_data:
        writer.writerow([num, label])
