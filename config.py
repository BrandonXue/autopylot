# ============== USER AND REMINDER ===============
CURRENT_USER = 'Brandon Xue'
# CURRENT_USER = 'Jacob Rapmund'
print('\nCar game config loaded for', CURRENT_USER, end='\n\n') # Remind on load

# =================== VIEWPORT ===================
VIEWPORT_WIDTH = {
    'Brandon Xue': 1080, # 1080
    'Jacob Rapmund': 1080
}[CURRENT_USER]

VIEWPORT_HEIGHT = {
    'Brandon Xue': 720, # 720
    'Jacob Rapmund': 720
}[CURRENT_USER]

VIEWPORT_SIZE = (VIEWPORT_WIDTH, VIEWPORT_HEIGHT)

# ====================== MAP =====================
MAP_WIDTH = {
    'Brandon Xue': 4000,
    'Jacob Rapmund': 4000
}[CURRENT_USER]

MAP_HEIGHT = {
    'Brandon Xue': 4000,
    'Jacob Rapmund': 4000
}[CURRENT_USER]

MAX_BOX_WIDTH = {
    'Brandon Xue': 100,
    'Jacob Rapmund': 100
}[CURRENT_USER]

MAX_BOX_HEIGHT = {
    'Brandon Xue': 100,
    'Jacob Rapmund': 100
}[CURRENT_USER]

MIN_BOX_WIDTH = {
    'Brandon Xue': 100,
    'Jacob Rapmund': 100
}[CURRENT_USER]

MIN_BOX_HEIGHT = {
    'Brandon Xue': 100,
    'Jacob Rapmund': 100
}[CURRENT_USER]

MAX_X_LOC_BOX = {
    'Brandon Xue': 1500,
    'Jacob Rapmund': 1500
}[CURRENT_USER]

MAX_Y_LOC_BOX = {
    'Brandon Xue': 1500,
    'Jacob Rapmund': 1500
}[CURRENT_USER]

MAP_SIZE = (MAP_WIDTH, MAP_HEIGHT)

# ====================== CAR =====================
CAR_WIDTH = {
    'Brandon Xue': 20,
    'Jacob Rapmund': 20
}[CURRENT_USER]

CAR_HEIGHT = {
    'Brandon Xue': 40,
    'Jacob Rapmund': 40
}[CURRENT_USER]

CAR_SIZE = (CAR_WIDTH, CAR_HEIGHT)

CAR_ACCELERATION = {
    'Brandon Xue': 0.05,
    'Jacob Rapmund': 0.05 #0.002
}[CURRENT_USER]

CAR_MAX_VELOCITY = {
    'Brandon Xue': 20.0, # 2.0
    'Jacob Rapmund': 20.0 #0.8
}[CURRENT_USER]

CAR_VELOCITY_ZERO_THRESHOLD = {
    'Brandon Xue': 0.01,
    'Jacob Rapmund': 0.01 #0.01
}[CURRENT_USER]

CAR_VELOCITY_DECAY = {
    'Brandon Xue': 0.95, # .99
    'Jacob Rapmund': 0.95 #0.993
}[CURRENT_USER]

CAR_ANGULAR_ACCELERATION = {
    'Brandon Xue': 0.01,
    'Jacob Rapmund': 0.004 #0.001
}[CURRENT_USER]

CAR_MAX_ANGULAR_VELOCITY = {
    'Brandon Xue': 0.5, # 0.4
    'Jacob Rapmund': 0.4 #0.4
}[CURRENT_USER]

CAR_ANGULAR_ZERO_THRESHOLD = {
    'Brandon Xue': 0.001,
    'Jacob Rapmund': 0.001 #0.001
}[CURRENT_USER]

CAR_ANGULAR_VELOCITY_DECAY = {
    'Brandon Xue': 0.95,
    'Jacob Rapmund': 0.95 #0.990
}[CURRENT_USER]

# ===================== OTHER ====================
KEY_ARRAY_SIZE = {
    'Brandon Xue': 1024,
    'Jacob Rapmund': 1024
}[CURRENT_USER]

FRAME_RATE = {
    'Brandon Xue': 60,
    'Jacob Rapmund': 120
}[CURRENT_USER]

GRAYSCALE_DIM = {
    'Brandon Xue': 80,
    'Jacob Rapmund': 80
}[CURRENT_USER]

NUMBER_OF_DOTS = {
    'Brandon Xue': 20,
    'Jacob Rapmund': 20
}[CURRENT_USER]