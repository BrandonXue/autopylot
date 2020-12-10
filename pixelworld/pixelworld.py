# Local modules
from pixelmaps import Map

# Non-local modules
from pygame import draw, Rect, surfarray
import pygame.locals

def clamp(value: float, min: float, max: float) -> float:
    if value > max:
        return max
    elif value < min:
        return min
    return value

class Entity:
    def __init__(self, x, y, size, color, world_bounds: Rect) -> None:
        self.__x = x
        self.__y = y
        self.__size = size
        self.color = color
        self.__bounds = world_bounds
        
    def get_x(self)->int:
        return self.__x

    def get_y(self)->int:
        return self.__y

    def move_x(self, spaces: int=1) -> None:
        self.__x += spaces
        self.__x = clamp(self.__x, self.__bounds.left, self.__bounds.right)

    def move_y(self, spaces: int=1) -> None:
        self.__y += spaces
        self.__y = clamp(self.__y, self.__bounds.top, self.__bounds.bottom)

    def draw(self, surface: pygame.Surface) -> None:
        draw.rect(
            surface, self.color, 
            Rect(self.__x * self.__size, self.__y * self.__size, self.__size, self.__size)
        )

    def draw_nn(self, surface: pygame.Surface) -> None:
        draw.rect(
            surface, self.color,
            Rect(self.__x, self.__y, self.__size, self.__size)
        )

class PixelWorld:
    # Entity Types
    EMPTY = 0
    PIT = 1
    PELLET = 2
    PLAYER = 3

    # Entity Colors
    PIT_COLOR = (0, 0, 0)
    PELLET_COLOR = (0, 0, 255)
    PLAYER_COLOR = (0, 255, 0)
    EMPTY_COLOR = (255, 255, 255)

    def __init__(self, game_map: Map, scale: float=32.0) -> None:
        ''' 
        unscaled_dims: raw pixel dimensions of the world before render scaling (width, height)
        '''

        # Logical units
        self.__world_bounds = Rect(0, 0, game_map.width, game_map.height)
        self.data_surf = pygame.Surface( (game_map.width, game_map.height) )

        # Display units
        self.scale = scale # For scaling up display
        self.view_width = int(game_map.width * self.scale)
        self.view_height = int(game_map.height * self.scale)
        self.game_surf = pygame.Surface( (self.view_width, self.view_height) )
        self.center_pos = (game_map.width // 2, game_map.height // 2)

        self.view_surf = pygame.display.set_mode(
            (self.view_width + 300, self.view_height + 300), # width and height
            pygame.DOUBLEBUF | pygame.HWSURFACE # mode flags
        )
        self.view_surf.set_alpha(None)

        # Attributes
        self.map = game_map
        self.player = None
        self.pits = []
        self.pellets = []

        # Metrics
        self.__points = 0
        self.__done = False

    def set_map(self, map_: Map) -> None:
        '''
        Load in a Map with a two dimensional List and other stats.
        You will need to call reset() before you can interact with the world.
        '''

        # Set map if valid
        self.map = map_

    def __load_map_to_arrays(self) -> None:
        ''' Helper method to load self.map entities into their respective arrays. '''
        
        self.pits = []
        self.pellets = []
        for row in range(len(self.map.data)):
            for col in range(len(self.map.data[row])):
                x, y = col, row
                if self.map.data[row][col] == PixelWorld.PLAYER:
                    self.start_pos = (x, y)
                    self.player = Entity(x, y, self.scale, PixelWorld.PLAYER_COLOR, self.__world_bounds)
                elif self.map.data[row][col] == PixelWorld.PELLET:
                    self.pellets.append(Entity(x, y, self.scale, PixelWorld.PELLET_COLOR, self.__world_bounds))
                elif self.map.data[row][col] == PixelWorld.PIT:
                    self.pits.append(Entity(x, y, self.scale, PixelWorld.PIT_COLOR, self.__world_bounds))

    def reset(self):
        '''
        Reset the environment to the starting state. Returns state.

        Raises Exception if map has not been set.
        '''
        
        # Make sure this isn't called before map is loaded
        if not self.map:
            raise Exception("Tried to access map, but map wasn't set.")

        self.__load_map_to_arrays()
        
        # Reset metrics
        self.__points = 0
        self.__done = False

        return self.get_event_step()[0]

    def num_actions(self) -> int:
        ''' 
        Used for probing and diagnostic purposes if someone
        wants to use this environment for other purposes.
        ''' 

        return 4 # Left, Up, Right, Down

    def perform_action(self, action: int):
        '''
        Apply an action to the environment.
        Left = 0, Up = 1, Right = 2, Down = 3. Stay put = -1.

        Raises Exception if action could not be interpreted.
        '''

        if action == -1:    # NOTE: This is intended for human players
            pass
        elif action == 0:   # Move left
            self.player.move_x(-1)
        elif action == 1:   # Move up
            self.player.move_y(-1)
        elif action == 2:   # Move right
            self.player.move_x(1)
        elif action == 3:   # Move down
            self.player.move_y(1)
        else:
            raise Exception("The action inputted was invalid: " + str(action))

    def get_done(self):
        '''
        Useful for human players
        '''
        return self.__done

    def get_event_step(self, action=-1):
        ''' 
        Updates keras output state 
        and
        Returns feedback from the envrionment (state, reward, done, info)
        '''

        self.perform_action(action)

        # Update the data surface (state)
        self.data_surf.fill(PixelWorld.EMPTY_COLOR)
        plyr_x = self.player.get_x()
        plyr_y = self.player.get_y()
        x_offset = self.center_pos[0] - plyr_x
        y_offset = self.center_pos[1] - plyr_y

        reward = 0

        i = 0
        while i < len(self.pellets):
            pellet_x = self.pellets[i].get_x()
            pellet_y = self.pellets[i].get_y()
            if plyr_x == pellet_x and plyr_y == pellet_y:
                self.__points += 1
                reward = 1
                self.pellets.pop(i)
            else:
                self.data_surf.set_at((pellet_x + x_offset, pellet_y + y_offset), self.pellets[i].color)
            i += 1
        
        for pit in self.pits:
            pit_x = pit.get_x()
            pit_y = pit.get_y()
            if plyr_x == pit_x and plyr_y == pit_y:
                reward = -1
                self.__done = True
            else:
                self.data_surf.set_at((pit.get_x() + x_offset, pit.get_y() + y_offset), pit.color)
        
        self.data_surf.set_at(self.center_pos, self.player.color)
        
        # Return (state, reward, done, info)
        return (surfarray.array3d(self.data_surf), reward, self.__done, None)

    def render(self, show_data_surf=False) -> None:
        # Update the game surface
        self.game_surf.fill(PixelWorld.EMPTY_COLOR)
        for pellet in self.pellets:             # Draw pellets onto game surface
            pellet.draw(self.game_surf)
        for pit in self.pits:                   # Draw pits onto game surface
            pit.draw(self.game_surf)

        self.player.draw(self.game_surf)        # Draw player onto game surface
        
        # Update the view surface
        self.view_surf.fill(PixelWorld.EMPTY_COLOR)
        self.view_surf.blit(self.game_surf, (150, 150))  
        if show_data_surf: # Player doesn't need, only to see what AI sees
            self.view_surf.blit(self.data_surf, (0,0))
        pygame.display.flip()