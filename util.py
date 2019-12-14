"""
@author: developer
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) )


def plot_images( images ):
    fig = plt.figure(figsize=(10,10))
    for i in range(10):
        for j in range(10):
            plt.subplot(10,10,i*10+(j+1))
            plt.imshow(images[i*10+j],cmap="gray")
    plt.savefig("RBM.png")
