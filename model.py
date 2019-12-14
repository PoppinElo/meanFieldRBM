"""
@author: developer
"""

import numpy as np
from util import *


class RBM:
    def __init__( self, visible_size = 28*28, hidden_size = 144, temperature = 1, binary_visible = False, binary_hidden = True ):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.temperature = temperature
        
        self.weights = np.random.random( size = (visible_size,hidden_size) ) - .5
        self.visible_bias = np.random.random( size = visible_size ) - .5
        self.hidden_bias = np.random.random( size = hidden_size ) - .5
        
        self.binary_visible = binary_visible
        self.binary_hidden = binary_hidden
    
    
    def feedforward( self, visible_units ):
        h = sigmoid( ( np.matmul( visible_units, self.weights ) + self.hidden_bias ) / self.temperature )
        if( self.binary_hidden ):
            return ( h > np.random.uniform( size = self.hidden_size ) ).astype( np.int )
        return h
    
    
    def feedback( self, hidden_units ):
        v = sigmoid( ( np.matmul( hidden_units, self.weights.T ) + self.visible_bias ) / self.temperature )
        if( self.binary_visible ):
            return ( v > np.random.uniform( size = self.visible_size ) ).astype( np.int )
        return v
    
    
    def train( self, data_input, learning_rate = 0.01 ):
        num_input = len(data_input)
        Dweights = np.zeros( shape = (self.visible_size,self.hidden_size) )
        Dvisible_bias = np.zeros( shape = self.visible_size )
        Dhidden_bias = np.zeros( shape = self.hidden_size )
        
        n_step = 100
        # positive phase
        for t in range( n_step ):
            h = self.feedforward( data_input )
            Dweights += np.matmul( data_input.T, h ) / n_step
            Dhidden_bias += np.sum( h, axis = 0 ) / n_step
        Dvisible_bias += np.sum( data_input, axis = 0 )
        # negative phase
        for t in range( n_step ):
            v = self.feedback( h )
            h = self.feedforward( v )
            Dweights -= np.matmul( v.T, h ) / n_step
            Dvisible_bias -= np.sum( v, axis = 0 ) / n_step
            Dhidden_bias -= np.sum( h, axis = 0 ) / n_step
        # update
        self.weights += ( learning_rate / self.temperature ) * Dweights / num_input
        self.visible_bias += ( learning_rate / self.temperature ) * Dvisible_bias / num_input
        self.hidden_bias += ( learning_rate / self.temperature ) * Dhidden_bias / num_input

        
    def generate_visible( self, num_of_images = 100, hidden_units = None ):
        if( hidden_units == None ):
            hidden_units = np.random.random( size = ( num_of_images, self.hidden_size ) )
            if( self.binary_hidden ):
                hidden_units = np.round( hidden_units )
        n_step = 100
        for t in range(n_step):
            visible_units = self.feedback(hidden_units)
            hidden_units = self.feedforward(visible_units)
        return visible_units
        
        
        
        
        
        
        
