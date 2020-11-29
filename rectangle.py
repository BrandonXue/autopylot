import numpy as np
from numpy.lib.function_base import vectorize
import pygame

class Rectangle:
    def __init__(self, x=0.0, y=0.0, width=10, height=10):
        self.max_dim = width if width > height else height

        # Set position vectors
        self.corners = [
            pygame.math.Vector2(x, y),
            pygame.math.Vector2(x+width, y),
            pygame.math.Vector2(x+width, y+height),
            pygame.math.Vector2(x, y+height)
        ]
        self.center_vec = pygame.math.Vector2(x + width/2, y + height/2)

    def pivot_and_offset(self, pivot_point, degrees, view_adjust):
        # For each corner, find a new vector pointing from pivot_point to it,
        # and rotate that new vector.
        return (
            (self.corners[0] - pivot_point).rotate(degrees) + view_adjust,
            (self.corners[1] - pivot_point).rotate(degrees) + view_adjust,
            (self.corners[2] - pivot_point).rotate(degrees) + view_adjust,
            (self.corners[3] - pivot_point).rotate(degrees) + view_adjust
        )
  
    def move(self, move_values:np.array(2) = np.array([0.,0.])):
        self.corners[0] += pygame.math.Vector2(move_values[0], move_values[1])
        self.corners[1] += pygame.math.Vector2(move_values[0], move_values[1])
        self.corners[2] += pygame.math.Vector2(move_values[0], move_values[1])
        self.corners[3] += pygame.math.Vector2(move_values[0], move_values[1])
        self.center_vec += pygame.math.Vector2(move_values[0], move_values[1])

        
    def get_center(self):
        return self.center_vec

    def get_max_dim(self):
        return self.max_dim

class PlayerRectangle(Rectangle):
    def __init__(self, x=0, y=0, width=10, height=10):
        super().__init__(x, y, width, height) 

        # Set position and state
        # We want a position near the rear because cars pivot around rear axle
        self.rear_center_vec = pygame.math.Vector2(x + width/2, y + height * 0.8)
        self.radians = 0.0
        self.angular_vel = 0.0
        self.vel = 0.0
        self.collision = False

    def is_colliding(self):
        return self.collision

    def get_rear_center(self):
        return self.rear_center_vec

    def move(self, move_values:np.array(2) = np.array([0.,0.])):
        super().move(move_values)
        self.rear_center_vec += move_values

def lin_seg_intersection(p1, p2, p3, p4): #p is a point either tuple or list in format x, y. p1 and p2 together form a line and p3 and p4 together form a line
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p4[0] - p3[0], p4[1] - p3[1])
    odd1 = p3[0] - p1[0]
    odd2 = p1[1] - p3[1]

    numt = v2[1]*odd1 + v2[0]*odd2
    nums = odd2*v1[0] + odd1*v1[1]

    den = v1[0]*v2[1] - v1[1]*v2[0]
    
    if den == 0:
        return False

    t = numt / den
    s = nums / den
    if t >= 0 and t <= 1 and s >= 0 and s <= 1:
        return True

    return False
    
def check_collision(ob_coords1, ob_coords2):
    for i in range(0, 4):
        for j in range(0, 4):
            if lin_seg_intersection(ob_coords1[i-1], ob_coords1[i], ob_coords2[j-1], ob_coords2[j]): return True
    return False