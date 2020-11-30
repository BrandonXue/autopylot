from config import *
from defs import ExitSignalType

import cv2
import mss
import numpy as np

import tensorflow as tf
from tensorflow import keras

class DataProcessor:
    def __init__(self, read_conn, write_conn):
        self.read_conn = read_conn
        self.write_conn = write_conn

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
            if self.read_conn.poll():
                received = self.read_conn.recv()

                # If exit signal received, perform handshake and return
                if type(received) is ExitSignalType:
                    self.write_conn.send(ExitSignalType())
                    return True
                
            # TODO: Train neural net here
            pixels = np.array(sct.grab(monitor)).astype('float32')
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)
            pixels = cv2.resize(pixels, (GRAYSCALE_DIM, GRAYSCALE_DIM))# / 255
            #pixels = np.transpose(pixels.astype('float64'), (1,0,2))
            pixels = np.transpose(pixels, (1,0))
            
            #comment out if you dont want data sent
            self.write_conn.send(pixels)
