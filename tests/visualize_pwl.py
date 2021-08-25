# %%
import matplotlib.pyplot as plt
import struct

# dataset
dataset = []
with open("../data/dna_uint32", "rb") as f:
    # f.seek(0, 2)    # to end
    # size = f.tell()
    # f.seek(0, 0)
    size = struct.unpack("Q", f.read(8))   # dna_uint32: read size
    for _ in range(size[0]): # sizeof(uint32_t) = 4
        dataset.append(struct.unpack("I", f.read(4)))

# indexes
indexes = range(len(dataset))


# boundary_x, boundary_y
boundary_x = []
boundary_y = []
with open("../include/data/boundaries", "rb") as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    for _ in range(int(size/12)): # sizeof(uint32_t) + sizeof(uint64_t) = 4 + 8 = 12
        boundary_x.append(struct.unpack("I", f.read(4)))
        boundary_y.append(struct.unpack("Q", f.read(8)))
for i in range(len(boundary_y)):
    boundary_x[i] = boundary_x[i][0]
    boundary_y[i] = boundary_y[i][0]

plt.plot(dataset, indexes, 'b')
plt.plot(boundary_x, boundary_y, 'ro--', marker="x")
plt.show()

# %%
