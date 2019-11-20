#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 01:55:48 2019

@author: developer
"""

import numpy as np
import mnist
import matplotlib.pyplot as plt

from util import *
from model import *


# parameters

VISIBLE_SIZE = 28 * 28
HIDDEN_SIZE = 144
T = 1

BATCH_SIZE = 20
NUM_EPOCHS = 2000

def main():
    train_images = mnist.train_images()
    network = RBM( visible_size = VISIBLE_SIZE, hidden_size = HIDDEN_SIZE, temperature = T )
    
    for epoch in range(NUM_EPOCHS):
        idx = epoch * BATCH_SIZE
        while( idx + BATCH_SIZE >  60000 ):
            idx -= 60000
        images_input = np.reshape(train_images[ idx: idx + BATCH_SIZE ],(BATCH_SIZE,28*28)) / 255
        network.train( data_input = images_input )
        
    generated_images = np.reshape(network.generate_visible(),(100,28,28)) * 255
    plot_images(generated_images)
    
main()
