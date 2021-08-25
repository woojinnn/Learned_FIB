import random
import struct
import os.path

random_numbers = []

for _ in range(30):
    random_numbers.append(random.randint(0, 50))

random_numbers.sort()
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/dataset"), 'wb') as f:
    for random_number in random_numbers:
        f.write(struct.pack('I', random_number))

print(random_numbers)