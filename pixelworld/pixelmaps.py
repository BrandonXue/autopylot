# Non-local modules
import cv2

class Map:
    ''' A game map with map stats. '''
    def __init__(self, data, num_pits: int, num_pellets: int, num_player_spots: int) -> None:
        self.data = data
        self.height = len(data)
        self.width = len(data[0])
        self.dims = (self.width, self.height)

        self.num_pits = num_pits
        self.num_pellets = num_pellets
        self.num_player_spots = num_player_spots

class MapLoader:
    ''' Used to load and interpret various file formats into a Map object. '''

    @staticmethod
    def __has_extension(filepath: str, ext: str) -> bool:
        return filepath.rfind(ext) == (len(filepath) - len(ext))

    @staticmethod
    def load(filepath: str) -> Map:
        ''' 
        Load a file into a map.

        .txt: empty = 0, pit = 1, pellet = 2, player = 3, spaces ignored

        .png: empty = 0xFFFFFF, pit = 0x000000, pellet = 0x0000FF, player = 0x00FF00

        Raises Exception if map had any error loading.
        '''

        if MapLoader.__has_extension(filepath, '.png'):
            data, num_pits, num_pellets, num_player_spots = MapLoader.__load_png(filepath)
        elif MapLoader.__has_extension(filepath, '.txt'):
            data, num_pits, num_pellets, num_player_spots = MapLoader.__load_txt(filepath)
        else:
            raise Exception("Map format not supported. Supported types:\n\t.png\t.txt")

        return Map(data, num_pits, num_pellets, num_player_spots)

    @staticmethod
    def __check_dimensions(data) -> None:
        ''' Make sure the given 2D List is not empty and rows have same length.'''

        if len(data) == 0:
            raise Exception("The map source was empty.")
        cols = len(data[0])
        if not all (len(row) == cols for row in data):
            raise Exception("The map source had uneven rows.")

    @staticmethod    
    def __load_png(filepath: str):
        ''' Load a .png into a Map'''

        data = []
        num_pits = 0
        num_pellets = 0
        num_player_spots = 0

        map_png = cv2.imread(filepath)                  # BGR Format
        if type(map_png) == type(None):
            raise Exception("The .png file was not found.")

        height, width, channels = map_png.shape
        if channels != 3:
            raise Exception(
                "Images with .png extension need BGR channels.\n"
                f"Found channels: {channels}."
            )

        for row in range(height):
            row_data = []
            for col in range(width):
                pixel = map_png[row][col]
                if (pixel == [255, 255, 255]).all():    # BGR White
                    row_data.append(0)
                elif (pixel == [0, 0, 0]).all():        # BGR black
                    num_pits += 1
                    row_data.append(1)
                elif (pixel == [255, 0, 0]).all():      # BGR blue
                    num_pellets += 1
                    row_data.append(2)
                elif (pixel == [0, 255, 0]).all():      # BGR green
                    num_player_spots += 1
                    row_data.append(3)
                else:
                    raise Exception(
                        f"Undefined pixel at row {str(row)} column {str(col)}.\n"
                        f"Found pixel (shown in BGR format): {map_png[row][col]}.\n"
                        "Check that cv2.imread is indeed reading as BGR format."
                    )
            data.append(row_data)

        MapLoader.__check_dimensions(data)
        
        return data, num_pits, num_pellets, num_player_spots
        
    @staticmethod
    def __load_txt(filepath: str):
        ''' Load a .txt file into a Map. '''

        data = []
        num_pits = 0
        num_pellets = 0
        num_player_spots = 0

        with open(filepath) as map_txt:
            lines = map_txt.readlines()
            for line in range(len(lines)):
                row_data = []
                for col in range(len(lines[line])):
                    if lines[line][col] == '0':
                        row_data.append(0)
                    elif lines[line][col] == '1':
                        num_pits += 1
                        row_data.append(1)
                    elif lines[line][col] == '2':
                        row_data.append(2)
                        num_pellets += 1
                    elif lines[line][col] == '3':
                        row_data.append(3)
                        num_player_spots += 1
                    elif lines[line][col] in {' ', '\n'}: # Ignore whitespace
                        pass
                    else:
                        raise Exception(
                            f"Undefined character at line {str(line+1)} column {str(col+1)}.\n"
                            f"Character: {lines[line][col]}."
                        )
                if len(row_data) > 0: # Ignore empty rows
                    data.append(row_data)

        MapLoader.__check_dimensions(data)

        return data, num_pits, num_pellets, num_player_spots