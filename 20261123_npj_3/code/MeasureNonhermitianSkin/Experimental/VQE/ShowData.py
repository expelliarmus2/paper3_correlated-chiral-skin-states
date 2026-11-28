import h5py
import numpy as np
import matplotlib.pyplot as plt

def Plot1():
    with h5py.File('optimization_results.h5', 'r') as f:
        for key in f.keys():
            plt.plot(f[key][:], label=key)
        plt.legend()
        plt.show()


if __name__=="__main__":
    Plot1()