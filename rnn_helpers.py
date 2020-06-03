import string
import random
import time
import math
import torch
import numpy as np
# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    with open (filename,"rb") as f:
        bytez = f.read()
    file = np.frombuffer(bytez,dtype=np.uint8)[np.newaxis,:][0]
    return file, len(file), bytez

# Turning a string into a tensor

def char_tensor(string):
    return torch.as_tensor(string)

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
