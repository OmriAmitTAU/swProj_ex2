import sys
import pandas as pd
import numpy as np
import mykmeanssp

vec_size = [-1]
centroids = []
centroids_index = []


class InputError(Exception):
    def __init__(self, cause=""):
        self.cause = cause

    def __str__(self):
        if self.cause == "k":
            return "Invalid number of clusters!"
        elif self.cause == "max_iter":
            return "Invalid maximum iteration!"
        else:
            return "An Error Has Occurred"


def euclidean_distance(vec1, vec2):
    d = 0
    for x1, x2 in zip(vec1, vec2):
        d_single = (x1 - x2) ** 2
        d += d_single
    return d ** 0.5


def to_float(num, arg_name):
    try:
        return float(num)
    except:
        raise InputError(arg_name)


def is_natural(num):
    return num > 0 and num - int(num) == 0


def init_vector_list(df_1, df_2):
    merged_df = pd.merge(df_1, df_2, left_on=df_1.columns[0], right_on=df_2.columns[0], how="inner", sort=True)
    merged_df = merged_df.set_index(merged_df.columns[0])
    return merged_df.values


def parse_input():
    if 5 <= len(sys.argv) <= 6:
        k = to_float(sys.argv[1], "k")
        max_iter = to_float(sys.argv[2], "max_iter") if len(sys.argv) == 6 else 300
        e = to_float(sys.argv[-3], "e")

        path_1 = pd.read_csv(sys.argv[-2], sep=',', header=None)
        path_2 = pd.read_csv(sys.argv[-1], sep=',', header=None)
        vectors_array = init_vector_list(path_1, path_2)

        n = vectors_array.shape[0]
        d = vectors_array.shape[1]

        if not assert_input(k, max_iter, n):
            return
        return int(k), int(n), int(d), int(max_iter), e, vectors_array
    else:
        raise Exception()


def assert_input(k, max_iter, n):
    if not (n > 1 and is_natural(n)):
        raise InputError()
    if not (1 < k < n and is_natural(k)):
        raise InputError("k")
    if not (1 < max_iter < 1000 and is_natural(max_iter)):
        raise InputError("max_iter")
    return True


def closest_centroid_distance(curr_vec):
    closest_d = float("inf")

    for centroid in centroids:
        distance = euclidean_distance(curr_vec, centroid)
        if distance < closest_d:
            closest_d = distance
    return closest_d


def print_first_line():
    print(','.join(map(str, centroids_index)))


def print_centroids(final_centroids):
    for final_centroid in final_centroids:
        centroid_str = ','.join('{:.4f}'.format(coord) for coord in final_centroid)
        print(centroid_str)


def setup_kmeans(k, n, vectors):
    marked = [False] * n
    np.random.seed(0)

    first_index = np.random.choice(n)
    marked[first_index] = True
    centroids_index.append(first_index)
    centroids.append(vectors[first_index])

    for _ in range(k - 1):
        distances = [closest_centroid_distance(vectors[i]) if not marked[i] else 0 for i in range(n)]
        probability = [d / sum(distances) for d in distances]
        curr_index = np.random.choice(n, p=probability)

        centroids_index.append(curr_index)
        centroids.append(vectors[curr_index])
        marked[curr_index] = True


def main():
    try:
        k, n, d, max_iter, epsilon, vectors_array = parse_input()
        setup_kmeans(k, n, vectors_array)
        vectors = vectors_array.tolist()
        curr_centroids = [centroids[i].tolist() for i in range(k)]

        new_centroids = mykmeanssp.fit(k, n, d, max_iter, epsilon, vectors, curr_centroids)
        print_first_line()
        print_centroids(new_centroids)
        print('')

    except InputError as e:
        print(e)
    except Exception:
        print("An Error Has Occurred")


if __name__ == "__main__":
    main()
