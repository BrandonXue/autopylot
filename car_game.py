from config import *
from defs import *
from car_details_menu import *
from rectangle import *

import pygame

from math import sqrt, radians, sin, cos
from random import randint
    
def clamp(value, min, max):
    if value > max:
        value = max
    elif value < min:
        value = min
    return value

class CarGame:
    def __init__(self, read_conn, write_conn):
        # We may want to init() selective modules (to look into if we have time)
        pygame.init()

        # Set inter-process communication properties
        self.read_conn = read_conn
        self.write_conn = write_conn

        # Game environment related
        self.world_bounds = Rectangle(-MAX_X_LOC_BOX, -MAX_Y_LOC_BOX, MAP_WIDTH, MAP_HEIGHT, RGB_WHITE)
        self.points = 0

        # Keyboard related
        self.key_list = [False] * KEY_ARRAY_SIZE # Lists are faster for simple random access

        # Display related
        self.display = pygame.display.set_mode(
            VIEWPORT_SIZE,
            pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.FULLSCREEN
        )
        self.display.set_alpha(None)

        self.cover_display = pygame.Surface((GRAYSCALE_DIM, GRAYSCALE_DIM))

        self.font = pygame.freetype.Font(None, 32)
        self.car_details_menu = CarDetailsMenu((300, 500), (5, 5), (150, 150, 150, 200))
    
        # Initial car state
        self.rotation = 0.0
        self.angular_velocity = 0.0 # Can think of this as steering wheel position
        self.velocity = 0.0
        self.collision = False

        # Pre-calculated game constants 
        self.half_viewport = pygame.math.Vector2(VIEWPORT_SIZE)/2
        self.display_offset = self.half_viewport
        self.display_diag = sqrt(self.half_viewport[0]**2 + self.half_viewport[1]**2)

        self.collision_check_dist = (
            sqrt((CAR_WIDTH/2)**2 + (CAR_HEIGHT/2)**2) # player/car box hypoetenuse
            + sqrt((MAX_BOX_WIDTH/2)**2 + (MAX_BOX_HEIGHT/2)**2) # terrain object max hypotenuse
            + 20 # Arbitrary number to ensure we don't filter out anything that looks like it should be colliding
        )

        # Frame counter for throttling certain events
        # self.frame_count = 0
        
    def draw_keras_view(self, pixels):
        self.cover_display.fill(RGB_WHITE)
        pygame.surfarray.blit_array(self.cover_display, pixels) 
        
    def load_map_from_file(self, filepath):
        in_file = open(filepath) #default read only
        
        i = 0
        
        self.rects = []
        self.reward_dot_pool = []
        
        for line in in_file.readlines():
            j = 0
            for object_code in line:
                if object_code == '0':
                    self.reward_dot_pool.append((i, j))
                elif object_code == '1':
                    self.rects.append(
                        EnvironmentRectangle(
                            i*MAX_BOX_WIDTH, j*MAX_BOX_HEIGHT, # Location coords
                            MAX_BOX_WIDTH, MAX_BOX_HEIGHT, RGB_BLACK # width height
                        )
                    )
                elif object_code == '2':
                    self.player_start_pos = (i, j)
                    self.player_rect = PlayerRectangle(
                        i*MAX_BOX_WIDTH + MAX_BOX_WIDTH/2, 
                        j*MAX_BOX_HEIGHT + MAX_BOX_HEIGHT/2, 
                        CAR_WIDTH, CAR_HEIGHT
                    )   
                j = j+1
            i = i+1
        in_file.close()
        self.add_goat_rects_from_pool()
    
    ##USE ONLY FOR RANDOM MAPS
    def create_bounds(self):
        self.bounds = []
        self.bounds.append(
            EnvironmentRectangle(
                -2000, -2000, # Location coords
                100, 100, RGB_BLACK # width height
            )
        )
        for i in range(1, 40):
            self.bounds.append(
                EnvironmentRectangle(
                    i*100-2000, -2000, # Location coords
                    100, 100, RGB_BLACK # width height
                )
            )
            self.bounds.append(
                EnvironmentRectangle(
                    39*100-2000, i*100-2000, # Location coords
                    100, 100, RGB_BLACK # width height
                )
            )
            self.bounds.append(
                EnvironmentRectangle(
                    i*100-2000, 39*100-2000, # Location coords
                    100, 100, RGB_BLACK # width height
                )
            )
            self.bounds.append(
                EnvironmentRectangle(
                    -2000, i*100-2000, # Location coords
                    100, 100, RGB_BLACK # width height
                )
            )
       
    def create_player_rect(self):
        self.player_rect = PlayerRectangle(0, 0, CAR_WIDTH, CAR_HEIGHT)
        
    def create_random_rects(self):
        self.rects = []
        for i in range(0, 50):
            self.rects.append(
                EnvironmentRectangle(
                    randint(-MAX_X_LOC_BOX, MAX_X_LOC_BOX), randint(-MAX_Y_LOC_BOX, MAX_Y_LOC_BOX), # Location coords
                    randint(MIN_BOX_WIDTH, MAX_BOX_WIDTH), randint(MIN_BOX_HEIGHT, MAX_BOX_HEIGHT) # width height
                )
            )
        self.rects.append(
            EnvironmentRectangle(
                randint(-MAX_X_LOC_BOX, MAX_X_LOC_BOX), randint(-MAX_Y_LOC_BOX, MAX_Y_LOC_BOX), 
                25, 25, RGB_GOLD, GOAL
            )
        )
    ##END USE ONLY FOR RANDOM MAPS
    
    #function for updating goal (will be a rectangle)
    def udpate_goal_rect(self):
        self.rects.pop() #goal rect should always be last when using only one goal rect
        self.rects.append(
            EnvironmentRectangle(
                randint(-MAX_X_LOC_BOX, MAX_X_LOC_BOX), randint(-MAX_Y_LOC_BOX, MAX_Y_LOC_BOX), 
                25, 25, RGB_GOLD, GOAL
            )
        )

    def add_goat_rects_from_pool(self): #needs set alive function
        for i in range(0, NUMBER_OF_DOTS):
            location = self.reward_dot_pool[randint(0, len(self.reward_dot_pool))]
            self.reward_dot_pool.remove(location)
            self.rects.append(
                EnvironmentRectangle(
                    location[0]*MAX_BOX_WIDTH + MAX_BOX_WIDTH/2, location[1]*MAX_BOX_HEIGHT + MAX_BOX_HEIGHT/2, 
                    25, 25, RGB_GOLD, GOAL
                )
            )
    
    def remove_goal_rect(self, goal_rect):
        self.goal_rects.remove(goal_rect)
    
    def store_input_keys(self):
        for event in pygame.event.get():
            # Treat quit event as escape button pressed
            if event.type == pygame.QUIT:
                self.key_list[pygame.K_ESCAPE] = True
            
            #MENU FUNCTIONS
            if event.type == pygame.MOUSEMOTION:
                self.car_details_menu.set_mouse_pos(pygame.mouse.get_pos())
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.car_details_menu.set_state(PRESS)
            if event.type == pygame.MOUSEBUTTONUP:
                self.car_details_menu.set_state(RELEASE)
                self.car_details_menu.set_state(CLICK)
            
            #VEHICLE FUNCTIONS
            elif event.type == pygame.KEYDOWN and event.key < KEY_ARRAY_SIZE:
                self.key_list[event.key] = True
            elif event.type == pygame.KEYUP and event.key < KEY_ARRAY_SIZE:
                self.key_list[event.key] = False

    def update_positions(self):
        pressed = False

        if self.key_list[pygame.K_w]:
            self.velocity -= self.car_details_menu.get_car_acceleration()#CAR_ACCELERATION
            pressed=True
            
        if self.key_list[pygame.K_s]:
            self.velocity += self.car_details_menu.get_car_acceleration()#CAR_ACCELERATION 
            pressed=True
            
        # Turning left only
        if self.key_list[pygame.K_a] and not self.key_list[pygame.K_d]:
            self.angular_velocity = clamp(
                self.angular_velocity + CAR_ANGULAR_ACCELERATION, # new value
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY # clamp bounds
            )

        # Turning right only 
        elif self.key_list[pygame.K_d] and not self.key_list[pygame.K_a]:
            self.angular_velocity = clamp(
                self.angular_velocity - CAR_ANGULAR_ACCELERATION, # new value
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY # clamp bounds
            )

        # Else either both pressed or neither pressed. Move wheel towards center.
        else:
            if (self.angular_velocity < CAR_ANGULAR_ZERO_THRESHOLD 
                and self.angular_velocity > -CAR_ANGULAR_ZERO_THRESHOLD):
                self.angular_velocity = 0.0
            if self.angular_velocity != 0.0:
                self.angular_velocity *= CAR_ANGULAR_VELOCITY_DECAY
            
        # Only turn if car is in motion
        if self.velocity != 0:
            # Make turn speed dependent on velocity
            self.rotation -= self.angular_velocity * self.velocity

        self.velocity = clamp(self.velocity, -CAR_MAX_VELOCITY, CAR_MAX_VELOCITY)
        
        if not pressed:
            if self.velocity < CAR_VELOCITY_ZERO_THRESHOLD and self.velocity > -CAR_VELOCITY_ZERO_THRESHOLD:
                self.velocity = 0.0
            if self.velocity != 0:
                self.velocity *= CAR_VELOCITY_DECAY
                
        pressed = False

        # Move player position, (you can add tuple to Vector2)
        rads = radians(self.rotation)
        self.player_rect.move( (sin(rads) * self.velocity, cos(rads) * self.velocity) )


    def draw_game_scene(self):
        # These locations are used as references
        player_center = self.player_rect.get_center()
        rear_axle = self.player_rect.get_rear_center()
        
        #Draw world boundaries and get coords
        world_coords = self.world_bounds.pivot_and_offset(rear_axle, self.rotation, self.display_offset)

        # Render the player
        player_coords = self.player_rect.pivot_and_offset(rear_axle, 0, self.display_offset)
        
        self.player_rect.draw(self.display, player_coords)

        self.collision = False
        
        #for obj in self.bounds:
        #
        #    obj_center = obj.get_center()
        #    max_viewable_dist = self.display_diag + obj.get_max_dim()
        #
        #    dist_between = player_center.distance_to(obj_center)
        #    if (dist_between < max_viewable_dist):
        #
        #        object_coords = obj.pivot_and_offset(rear_axle, self.rotation, self.display_offset)
        #        
        #        if (not self.collision and dist_between < self.collision_check_dist):
        #            if check_collision(player_coords, object_coords):
        #                self.collision = True
        #        
        #        obj.draw(self.display, object_coords)
        
        for obj in self.rects:
                
            if obj.get_is_alive():
                obj_center = obj.get_center()
                max_viewable_dist = self.display_diag + obj.get_max_dim()

                dist_between = player_center.distance_to(obj_center)
                if (dist_between < max_viewable_dist):

                    object_coords = obj.pivot_and_offset(rear_axle, self.rotation, self.display_offset)
                    
                    if (not self.collision and dist_between < self.collision_check_dist):
                        if check_collision(player_coords, object_coords):
                            self.collision = True
                            if obj.get_type() == GOAL:
                                self.points += 1
                                obj.set_is_alive(False)
                                #self.udpate_goal_rect()
                    
                    obj.draw(self.display, object_coords)

    def draw_dashboard(self):
        # self.frame_count = (self.frame_count + 1) % (FRAME_RATE//60)
        #if self.key_list[pygame.K_p] and self.frame_count == 0:
        #    pixels = pygame.surfarray.pixels3d(self.display)
        #    self.pipe_conn.send(pixels)
            
        if not self.cover_display.get_locked() and not self.display.get_locked():
            self.display.blit(self.cover_display, (VIEWPORT_WIDTH-180, VIEWPORT_HEIGHT-180))
        
        # temp_menu.draw(viewport)
        col_text = f"Collisions: {'True' if self.collision else 'False'}"
        point_text = f"Points: {self.points}"
        fps_text = "FPS: {:.2f}".format(self.fps)
        self.font.render_to(self.display, (5, 5), col_text, RGBA_LIGHT_RED)
        self.font.render_to(self.display, (5, 35), point_text, RGBA_LIGHT_RED)
        self.font.render_to(self.display, (5, VIEWPORT_HEIGHT-35), fps_text, RGBA_LIGHT_RED)

    

    def start(self):
        #self.create_bounds()
        #self.create_player_rect()
        #self.create_random_rects()
        
        self.load_map_from_file("GameMap_1.txt")
        
        self.clock = pygame.time.Clock()
        
        while True:
            # Mange framerate
            self.clock.tick(FRAME_RATE)
            #self.clock.tick_busy_loop(FRAME_RATE) #This version will use more CPU
            self.fps = self.clock.get_fps()

            # Get input events
            self.store_input_keys()

            # If escape is pressed, break main event loop
            if self.key_list[pygame.K_ESCAPE]:
                # Send exit signal and wait for handshake 
                self.write_conn.send(ExitSignalType())
                while not type(self.read_conn.recv()) == ExitSignalType: pass
                return True

            # Fill background color
            self.display.fill(RGB_WHITE)

            # Draw player, rectangles, etc.
            self.draw_game_scene()
            # Update player position
            self.update_positions()

            # See if there are any updates from data processor
            if self.read_conn.poll():
                reply = self.read_conn.recv()
                self.draw_keras_view(reply)

            self.draw_dashboard()

            # Swap buffers
            pygame.display.flip()