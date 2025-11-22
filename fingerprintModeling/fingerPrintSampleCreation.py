import random
import csv

random_array = {}
setup_array = [1, 4, 2, 7, 9, 0, 3, 4]
num_array = {}
binary_array = {}

for i in range(10000):
    random_array[i] = random.randint(0, 9)

for i in range(10000):
    setup_array[0] = setup_array[1]
    setup_array[1] = setup_array[2]
    setup_array[2] = setup_array[3]
    setup_array[3] = setup_array[4]
    setup_array[4] = setup_array[5]
    setup_array[5] = setup_array[6]
    setup_array[6] = setup_array[7]
    setup_array[7] = random_array[i]
    num_array[i] = setup_array[0]*10000000 + setup_array[1]*1000000 + setup_array[2]*100000 + setup_array[3]*10000 + setup_array[4]*1000 + setup_array[5]*100 + setup_array[6]*10 + setup_array[7]
    if num_array[i] % 32 == 0:
        binary_array[i] = 1
    else:
        binary_array[i] = 0
    print(f'Number:{num_array[i]} Binary:{binary_array[i]}')


with open('output_32.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['data', 'label'])  # Header
    for i in range(10000):
        writer.writerow([num_array[i], binary_array[i]])