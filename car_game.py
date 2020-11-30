from config import *
from defs import *
from car_details_menu import *
from rectangle import *

import pygame

from math import sqrt, radians, sin, cos
from random import randint
from skimage import color, transform, exposure
    
def clamp(value, min, max):
    if value > max:
        value = max
    elif value < min:
        value = min
    return value

class CarGame:
    def __init__(self, pipe_conn, exit_signal):
        # We may want to init() selective modules (to look into if we have time)
        pygame.init()

        # Set inter-process communication properties
        self.pipe_conn = pipe_conn
        self.exit_signal = exit_signal
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

        # View transform related 
        # self.view_pos = np.array([0., 0.]) # viewport position needs to center on object

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
        self.frame_count = 0
        
    def update_keras_view(self, pixels):
        self.cover_display.fill(RGB_WHITE)
        pygame.surfarray.blit_array(self.cover_display, pixels)
        
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
        
    #function for updating goal (will be a rectangle)
    def udpate_goal_rect(self):
        self.rects.pop() #goal rect should always be last
        self.rects.append(
            EnvironmentRectangle(
                randint(-MAX_X_LOC_BOX, MAX_X_LOC_BOX), randint(-MAX_Y_LOC_BOX, MAX_Y_LOC_BOX), 
                25, 25, RGB_GOLD, GOAL
            )
        )

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
        move_vec = pygame.math.Vector2(0.0, 0.0)

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

        rads = radians(self.rotation)
        move_vec[0] += sin(rads) * self.velocity
        move_vec[1] += cos(rads) * self.velocity
        
        if not pressed:
            if self.velocity < CAR_VELOCITY_ZERO_THRESHOLD and self.velocity > -CAR_VELOCITY_ZERO_THRESHOLD:
                self.velocity = 0.0
            if self.velocity != 0:
                self.velocity *= CAR_VELOCITY_DECAY
                
        pressed = False

        # Move player position based on movement vector
        self.player_rect.move(move_vec)

    def render_updates(self):
        # Fill background color
        self.display.fill(RGB_WHITE)
        
        # These locations are used as references
        player_center = self.player_rect.get_center()
        rear_axle = self.player_rect.get_rear_center()
        
        #Draw world boundaries and get coords
        world_coords = self.world_bounds.pivot_and_offset(rear_axle, self.rotation, self.display_offset)

        # Render the player
        player_coords = self.player_rect.pivot_and_offset(rear_axle, 0, self.display_offset)
        
        self.player_rect.draw(self.display, player_coords)

        collision = False
        
        for obj in self.bounds:

            obj_center = obj.get_center()
            max_viewable_dist = self.display_diag + obj.get_max_dim()

            dist_between = player_center.distance_to(obj_center)
            if (dist_between < max_viewable_dist):

                object_coords = obj.pivot_and_offset(rear_axle, self.rotation, self.display_offset)
                
                if (not collision and dist_between < self.collision_check_dist):
                    if check_collision(player_coords, object_coords):
                        collision = True
                
                obj.draw(self.display, object_coords)
        
        for obj in self.rects:

            obj_center = obj.get_center()
            max_viewable_dist = self.display_diag + obj.get_max_dim()

            dist_between = player_center.distance_to(obj_center)
            if (dist_between < max_viewable_dist):

                object_coords = obj.pivot_and_offset(rear_axle, self.rotation, self.display_offset)
                
                if (not collision and dist_between < self.collision_check_dist):
                    if check_collision(player_coords, object_coords):
                        collision = True
                        if obj.get_type() == GOAL:
                            self.points += 1
                            self.udpate_goal_rect()
                
                obj.draw(self.display, object_coords)
        
        # ============== After scene is drawn, but before overhead display ===============
        self.frame_count = (self.frame_count + 1) % (FRAME_RATE//60)
        #if self.key_list[pygame.K_p] and self.frame_count == 0:
        #    pixels = pygame.surfarray.pixels3d(self.display)
        #    self.pipe_conn.send(pixels)
            
        if not self.cover_display.get_locked() and not self.display.get_locked():    
            self.display.blit(self.cover_display, (VIEWPORT_SIZE[0]-180,VIEWPORT_SIZE[1]-180))
        
        # ============= After input buffer is used for neural net draw HUD ===============
        # temp_menu.draw(viewport)
        col_text = f"Collisions: {'True' if collision else 'False'}"
        point_text = f"Points: {self.points}"
        fps_text = "FPS: {:.2f}".format(self.fps)
        self.font.render_to(self.display, (5, 5), col_text, RGBA_LIGHT_RED)
        self.font.render_to(self.display, (5, 35), point_text, RGBA_LIGHT_RED)
        self.font.render_to(self.display, (5, VIEWPORT_SIZE[1]-35), fps_text, RGBA_LIGHT_RED)

        # ========================= Swap the buffer to display ===========================
        #swap buffers
        pygame.display.flip()

    def start(self):
        self.create_bounds()
        self.create_player_rect()
        self.create_random_rects()
        
        self.clock = pygame.time.Clock()
        
        while True:
            self.store_input_keys()
            
            #self.clock.tick()
            self.clock.tick(FRAME_RATE)
            #self.clock.tick_busy_loop(FRAME_RATE) #This version will use more CPU
            self.fps = self.clock.get_fps()

            # If escape is pressed, break main event loop
            if self.key_list[pygame.K_ESCAPE]:
                self.pipe_conn.send(self.exit_signal)
                break
                
            if self.pipe_conn.poll():
                reply = self.pipe_conn.recv()
                self.update_keras_view(reply)

            self.update_positions()
            self.render_updates()
            
            