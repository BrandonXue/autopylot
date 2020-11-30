from config import *
from defs import ExitSignalType

from skimage import color, transform, exposure

import cv2
import dill
import mss
import numpy as np

import tensorflow as tf
from tensorflow import keras

class DataProcessor:
    def __init__(self, pipe_conn, exit_signal):
        self.pipe_conn = pipe_conn
        self.exit_signal = exit_signal

    def close_pipe(self):
        self.pipe_conn.close()

    def start(self):
            
        sct = mss.mss()
        mon = sct.monitors[1]
        
        monitor = {
            "top": mon["top"],
            "left": mon["left"],
            "width": VIEWPORT_WIDTH,
            "height": VIEWPORT_HEIGHT,
            "mon": 1,
        }
        
        dense = keras.layers.Dense(units=16)
        inputs = keras.Input(shape=(GRAYSCALE_DIM, GRAYSCALE_DIM, 3))
        
            
        while True:
            if self.pipe_conn.poll():
                received = self.pipe_conn.recv()
                if type(received) is ExitSignalType:
                    break
                
            # TODO: Train neural net here
            pixels = np.array(sct.grab(monitor)).astype('float32')
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)
            pixels = cv2.resize(pixels, (GRAYSCALE_DIM, GRAYSCALE_DIM)) / 255
            #pixels = np.transpose(pixels.astype('float64'), (1,0,2))
            pixels = np.transpose(pixels, (1,0))
            
            
            
            if dill.pickles(pixels):
                self.pipe_conn.send(pixels)
