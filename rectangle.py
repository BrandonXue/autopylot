from pygame.math import Vector2
from pygame.draw import polygon
from defs import *

#Environment types
OBSTACLE=0
GOAL=1
##################

class Rectangle:
    def __init__(self, x=0.0, y=0.0, width=10, height=10, color=RGB_BLACK):
        self.max_dim = width if width > height else height

        # Set position vectors
        self.corners = [
            Vector2(x, y),
            Vector2(x+width, y),
            Vector2(x+width, y+height),
            Vector2(x, y+height)
        ]
        self.center_vec = Vector2(x + width/2, y + height/2)
        self.color_ = color

    def pivot_and_offset(self, pivot_point, degrees, view_adjust):
        # For each corner, find a new vector pointing from pivot_point to it,
        # and rotate that new vector.
        return (
            (self.corners[0] - pivot_point).rotate(degrees) + view_adjust,
            (self.corners[1] - pivot_point).rotate(degrees) + view_adjust,
            (self.corners[2] - pivot_point).rotate(degrees) + view_adjust,
            (self.corners[3] - pivot_point).rotate(degrees) + view_adjust
        )
  
    def move(self, move_vec: Vector2 = Vector2(0.0, 0.0)):
        self.corners[0] += move_vec
        self.corners[1] += move_vec
        self.corners[2] += move_vec
        self.corners[3] += move_vec
        self.center_vec += move_vec

    def get_center(self):
        return self.center_vec

    def get_max_dim(self):
        return self.max_dim
        
    def draw(self, surface, coordinates):
        polygon(surface, self.color_, coordinates)
        
class EnvironmentRectangle(Rectangle):
    def __init__(self, x=0, y=0, width=10, height=10, color=RGB_GREEN, type=OBSTACLE):
        super().__init__(x, y, width, height, color)
        self.type_ = type
        self.is_alive = True
        
    def get_type(self):
        return self.type_
        
    def set_is_alive(self, is_alive):
        self.is_alive = is_alive
        
    def get_is_alive(self):
        return self.is_alive

class PlayerRectangle(Rectangle):
    def __init__(self, x=0, y=0, width=10, height=10):
        super().__init__(x, y, width, height, RGB_BLUE) 

        # Set position and state
        # We want a position near the rear because cars pivot around rear axle
        self.rear_center_vec = Vector2(x + width/2, y + height * 0.8)

    def is_colliding(self):
        return self.collision

    def get_rear_center(self):
        return self.rear_center_vec

    def move(self, move_vec: Vector2 = Vector2(0.0, 0.0)):
        super().move(move_vec)
        self.rear_center_vec += move_vec

def lin_seg_intersection(p1, p2, p3, p4): #p is a point either tuple or list in format x, y. p1 and p2 together form a line and p3 and p4 together form a line
    v1 = p2 - p1
    v2 = p4 - p3

    odd = Vector2(p1[1] - p3[1], p3[0] - p1[0])
    numt = odd.dot(v2)
    nums = odd.dot(v1)

    den = v1[0]*v2[1] - v1[1]*v2[0]
    
    if den == 0:
        return False

    t = numt / den
    s = nums / den
    if t >= 0 and t <= 1 and s >= 0 and s <= 1:
        return True

    return False
    
def check_collision(ob_coords1, ob_coords2):
    for i in range(4):
        for j in range(4):
            if lin_seg_intersection(ob_coords1[i-1], ob_coords1[i], ob_coords2[j-1], ob_coords2[j]): return True
    return False