# %%
import matplotlib.pyplot as plt
import struct

# dataset
dataset = []
with open("data/dataset", "rb") as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    for _ in range(int(size/4)): # sizeof(uint32_t) = 4
        dataset.append(struct.unpack("I", f.read(4)))

for i in range(len(dataset)):
    dataset[i] = dataset[i][0]

# indexes
indexes = range(len(dataset))

# boundary_y
boundary_y = []
with open("data/boundaries", "rb") as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    for _ in range(int(size/4)): # sizeof(uint32_t) = 4
        boundary_y.append(struct.unpack("I", f.read(4)))
for i in range(len(boundary_y)):
    boundary_y[i] = boundary_y[i][0]

# boundary_x
boundary_x = [dataset[i] for i in boundary_y]

plt.plot(dataset, indexes, 'b')
plt.plot(boundary_x, boundary_y, 'ro--', marker="x")
plt.show()

# %%
