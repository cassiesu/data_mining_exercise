#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys

r = 16
b = 32
n = 200000


def generate_hash_functions(size):
    return [np.random.random_integers(1, 20000, 3) for _ in range(0, size)]

def generate_band_hash_functions(size):
    return [np.random.random_integers(1, 20000, size + 1) for _ in range(0, b)]

def compute_hash(hash_func, shingle):
    a = hash_func[0]
    b = hash_func[1]
    c = hash_func[2]
    return (a * shingle + b) % c

def compute_min_hash(hash_func, shingles):
    hashed_shingles = [compute_hash(hash_func, x) for x in shingles]
    return min(hashed_shingles)

def compute_band_hash(band, hash_function):
    sum = hash_function[len(band)]
    for i in range(0, len(band)):
        sum += band[i] * hash_function[i]
    return sum % n

def partition(video_id, shingles):
    #hash document
    shingles_hash = [compute_min_hash(hash_func=hf, shingles=shingles) for hf in hash_functions]
    signature_column = np.array(shingles_hash)
    for i in range(0, b):
        start_index = i * r
        end_index = start_index + r
        band = signature_column[start_index:end_index]
        band_hash = compute_band_hash(band, band_hash_functions[i])
        print "%s\t%s" % (band_hash, (video_id, shingles.tolist()))


if __name__ == "__main__":
    # Very important. Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)
    hash_functions = generate_hash_functions(r * b)
    band_hash_functions = generate_band_hash_functions(r)
    for line in sys.stdin:
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], sep=" ")
        partition(video_id, shingles)

