import timeit
import numpy as np

def foo():
    a = np.arange(100000)
    b = np.arange(100000)
    c = a + b
    return c

if __name__ == "__main__":
    # Measure the execution time of foo function
    execution_time = timeit.repeat(foo, number=100, repeat=5)

    # print mean, variance, min, max
    print(f"Mean execution time: {np.mean(execution_time)} seconds")
    print(f"Variance of execution time: {np.var(execution_time)} seconds")
    print(f"Minimum execution time: {np.min(execution_time)} seconds")
    print(f"Maximum execution time: {np.max(execution_time)} seconds")
