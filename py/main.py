#Importacion de librerias
import os
import sys
import cv2
import time
import pybullet as p
import pybullet_data
import numpy as np
from info_arucos import *
from utils import cvK2BulletP, get_img_cam, load_maze

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../herramientas')))
from aruco_huntingv2 import ArucoHunting
from moveomni import MoveOmni

# DEBUG
DEBUG = True
def debug_print(*args):
    if DEBUG: print(*args)

# Define the sequence of marker IDs to detect
marker_sequence =  [1, 301, 401, 402, 100, 100, 305, 100, 307, 200, 100, 406]

# Initialize the state dictionary
state = {
    'current_marker_index': 0,
    'phase': 'approach',
    'detected_markers': [],  # Keep track of detected marker IDs
    'rotation_count': 0,
    'search_mode': 'close'  # Start with 'close' search mode
}

# Initialize t_cam, t_mov, and img *outside* the functions
t_cam = 0         # Controla la frecuencia de la cámara
t_mov = 0         # Controla la frecuencia de movimiento
img = None        # Imagen de la cámara
next_pose = None  # Siguiente posición del robot

# Connect to the PyBullet simulator
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Set the search path to find the plane.urdf file

p.setGravity(0,0,-9.81) # Set the gravity to be in the negative z direction
p.setTimeStep(1 / 240)

p.loadURDF("plane.urdf") # Cargar el plano
load_maze() # Cargar el laberinto con arucos

# Cargar el robot omni a una altura de 0.08 m
omni_base_pos = [0, 0, 0.08]
omni_id = p.loadURDF("../modelos/mini_omni/urdf/mini_omni.xacro", basePosition = omni_base_pos, useFixedBase = True)

# Clase que controla las posiciones del robot
move_omni = MoveOmni([0, 0, 0], vel = 30)

# Matriz de la cámara
camera_matrix = np.array([[691.,0. , 289.],[0., 690., 264.], [0., 0., 1.]])

# Clase que detecta arucos
hunter = ArucoHunting()
hunter.set_marker_length(0.1) # Dado por cubo simulado
hunter.set_camera_parameters(camera_matrix)

# Aruco detection parameters for different distances
params_close = cv2.aruco.DetectorParameters()
params_close.adaptiveThreshConstant = 7  # Tune these values
params_close.minMarkerPerimeterRate = 0.03
params_close.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Use subpixel refinement

params_far = cv2.aruco.DetectorParameters()
params_far.adaptiveThreshConstant = 5  # Tune these values
params_far.minMarkerPerimeterRate = 0.01
params_far.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Use subpixel refinement

# Example marker positions (you'll need to populate this dictionary with actual marker locations)
marker_positions = {
    1: [0.5, 0.0],
    301: [0.5, 0.7],
    # ... all the other marker locations
}

# Funciones para actualizar la cámara y el movimiento del robot
def update_camera():
    """
    Función que actualiza la imagen de la cámara cada 0.01 segundos
    """
    global img
    global t_cam
    global move_omni

    # Cada 0.01 segundos se actualiza la imagen de la cámara
    if t_cam + 0.01 < time.time():
        position_omni = move_omni.act_pose
        theta_rad = position_omni[2]
        theta_deg = np.rad2deg(theta_rad)

        camorientation = np.array([theta_deg, 0, 0])
        camposition = np.concatenate([position_omni[:2], [0]]) + [0.05 * np.cos(theta_rad), 0.05 * np.sin(theta_rad), 0.1] # Posición de la cámara respecto al robot
        p.addUserDebugPoints([camposition], [[1, 0, 0]], pointSize=2, lifeTime=0.001)

        img, _, _ = get_img_cam(width=480,
                                height=480, 
                                camposition = camposition, 
                                camorientation = camorientation, 
                                cam_mat=camera_matrix)
        t_cam = time.time()
    return img

def update_mov():
    """
    Función que actualiza el movimiento del robot omni cada 0.01 segundos
    """
    global t_mov
    global next_pose
    global move_omni

     # Si no está en la posición objetivo
    if t_mov + 0.01 < time.time() and move_omni.is_on_target() == False:
        move_omni.update_pose()

        new_position = np.concatenate([move_omni.act_pose[:2], [0.08]])
        new_orientation = np.concatenate([ [0.0, 0.0], [move_omni.act_pose[2]]])
        new_orientation = p.getQuaternionFromEuler(new_orientation)

        p.addUserDebugPoints([new_position], [[0, 1, 0]], pointSize=2, lifeTime=0.001)
        p.resetBasePositionAndOrientation(omni_id, new_position, new_orientation)
        t_mov = time.time()

    # Si está en la posición objetivo
    if move_omni.is_on_target() == True:
        next_pose = None # Reiniciar la siguiente posición

def detect_aruco(image, params):
    """
    Detect ArUco markers in the current camera frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    detected_markers = []
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.1, camera_matrix, None)
        for i, marker_id in enumerate(ids.flatten()):
            detected_markers.append({
                'id': marker_id,
                'rvec': rvecs[i],
                'tvec': tvecs[i]
            })
        
        # Draw markers for visual confirmation
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        if DEBUG:
            cv2.imshow("Detected Markers", image)
            cv2.waitKey(1)
    
    return detected_markers

def calculate_next_position(marker_detected):
    """
    Calculate the next position based on the detected marker and current phase.
    """
    global state, move_omni
    robot_x, robot_y, robot_rz = move_omni.act_pose
    tvec = marker_detected['tvec'].reshape(3)

    # Camera to robot frame transformation
    t_camera_in_robot = np.array([0.05, 0.0, 0.1])  # [x_offset, y_offset, z_offset]

    # Transform tvec from camera frame to robot frame
    t_marker_in_robot_x = tvec[2] + t_camera_in_robot[0]
    t_marker_in_robot_y = -tvec[0] + t_camera_in_robot[1]

    t_marker_in_robot = np.array([t_marker_in_robot_x, t_marker_in_robot_y])

    # Transform marker position from robot frame to world frame
    cos_rz = np.cos(robot_rz)
    sin_rz = np.sin(robot_rz)
    R_robot_to_world = np.array([[cos_rz, -sin_rz],
                                [sin_rz, cos_rz]])

    marker_pos_world = np.array([robot_x, robot_y]) + R_robot_to_world.dot(t_marker_in_robot)

    if state['phase'] == 'approach':
        # Move towards a point before the marker to avoid collision
        approach_distance = 0.3  # Adjust as needed
        direction_to_marker = marker_pos_world - np.array([robot_x, robot_y])
        distance_to_marker = np.linalg.norm(direction_to_marker)
        if distance_to_marker > approach_distance:
            desired_position = marker_pos_world - (approach_distance / distance_to_marker) * direction_to_marker
            desired_rz = np.arctan2(direction_to_marker[1], direction_to_marker[0])
        else:
            # Switch to rotation phase when close enough
            state['phase'] = 'rotate'
            # Calculate rotation based on marker ID
            marker_id = marker_detected['id']
            debug_print(f"Detected marker ID: {marker_id}, switching to rotate phase.")
            if marker_id in [1, 301, 401, 305]:  # Left turn
                desired_rz = robot_rz + np.pi / 2
            elif marker_id in [100, 307]:  # Right turn
                desired_rz = robot_rz - np.pi / 2
            elif marker_id == 402:  # 180° turn
                desired_rz = robot_rz + np.pi
            elif marker_id == 406:  # 360° turn
                desired_rz = robot_rz + 2 * np.pi
            else:
                desired_rz = robot_rz
            desired_position = np.array([robot_x, robot_y])

        return [desired_position[0], desired_position[1], desired_rz]

    elif state['phase'] == 'rotate':
        # Calculate the target rotation based on the marker ID
        marker_id = marker_detected['id']
        target_rz = None # initialize target rotation

        if marker_id in [1, 301, 401, 305]:  # Left turn
            target_rz = robot_rz + np.pi / 2
        elif marker_id in [100, 307]:  # Right turn
            target_rz = robot_rz - np.pi / 2
        elif marker_id == 402:  # 180° turn
            target_rz = robot_rz + np.pi
        elif marker_id == 406:  # 360° turn
            target_rz = robot_rz + 2 * np.pi
        else:  # markers like 200
            target_rz = robot_rz # maintain current rotation

        debug_print(f"Rotating to target_rz: {np.rad2deg(target_rz)} degrees")

        # Check if rotation is complete
        if abs(robot_rz - target_rz) < np.deg2rad(5): # Using a tolerance (5 deg)
            state['phase'] = 'approach'
            state['current_marker_index'] = min(state['current_marker_index'] + 1, len(marker_sequence) - 1)

            # Move forward a bit after rotation to avoid re-detecting the same marker
            delta_x = 0.1 * np.cos(robot_rz)
            delta_y = 0.1 * np.sin(robot_rz)

            return [robot_x + delta_x, robot_y + delta_y, robot_rz]
        else: # Rotate towards the target rotation

            # Incremental Rotation
            increment = np.deg2rad(10) # 10 degrees per step
            rotation_direction = np.sign(target_rz - robot_rz)

            return [robot_x, robot_y, robot_rz + rotation_direction*increment]

def get_next_pose(img):
    """
    Process camera image, update navigation state, and calculate next position.
    """
    global state, marker_sequence, move_omni

    current_marker_id = marker_sequence[state['current_marker_index']]

    if current_marker_id in state['detected_markers']:
        if state['phase'] == 'rotate':
            # If already detected and in rotate phase, finish rotation
            return calculate_next_position(marker_detected) # Continue previous rotate

        return None  # Don't recalculate if already detected and not rotating

    # Estimate distance to next marker (replace with your actual distance estimation)
    robot_x, robot_y, _ = move_omni.act_pose
    target_marker_pos = marker_positions.get(current_marker_id) # returns None if marker_id does not exist in the dictionary

    if target_marker_pos is not None:
        distance = np.linalg.norm(np.array(target_marker_pos) - np.array([robot_x, robot_y]))
    else: # handle cases when current marker id is not in your predefined marker position dictionary
        distance = 0 # or some other default value

    # Choose parameters based on distance
    params = params_far if distance > 0.5 else params_close  # Adjust threshold as needed

    detected_markers = detect_aruco(img, params)
    marker_detected = next((m for m in detected_markers if m['id'] == current_marker_id), None)

    if marker_detected:
        state['detected_markers'].append(current_marker_id)  # Register successful detection
        state['rotation_count'] = 0
        state['search_mode'] = 'close'

        if state['phase'] == 'approach': # only update calculate_next_position when approaching
            return calculate_next_position(marker_detected)
        elif state['phase'] == 'rotate': # continue rotating even if marker already detected
            return calculate_next_position(marker_detected)

    else:  # Marker not detected
        state['rotation_count'] += 1  # increment rotation count when marker not detected

        if state['phase'] == 'rotate':
            # DO NOT USE marker_detected HERE because it's not defined if the marker wasn't detected.
            robot_x, robot_y, robot_rz = move_omni.act_pose
            target_rz = None  # Initialize target_rz

            # Get the ID of the marker we ARE trying to detect, even if not seen.
            current_marker_id = marker_sequence[state['current_marker_index']] 

            if current_marker_id in [1, 301, 401, 305]:  # Left turn (use current_marker_id)
                target_rz = robot_rz + np.pi / 2
            elif current_marker_id in [100, 307]:  # Right turn
                target_rz = robot_rz - np.pi / 2
            elif current_marker_id == 402:  # 180° turn
                target_rz = robot_rz + np.pi
            elif current_marker_id == 406:  # 360° turn
                target_rz = robot_rz + 2 * np.pi

            if target_rz is not None:  # Only proceed if target_rz has been set
                if abs(robot_rz - target_rz) < np.deg2rad(5):
                    # Rotation is complete
                    state['phase'] = 'approach'
                    state['current_marker_index'] = min(state['current_marker_index'] + 1, len(marker_sequence) - 1)

                    # Move forward a bit after rotation to avoid re-detecting the same marker
                    delta_x = 0.1 * np.cos(robot_rz)
                    delta_y = 0.1 * np.sin(robot_rz)

                    return [robot_x + delta_x, robot_y + delta_y, robot_rz]
                else:
                    # Incremental rotation towards target_rz
                    increment = np.deg2rad(10)  # 10 degrees per step
                    rotation_direction = np.sign(target_rz - robot_rz)

                    return [robot_x, robot_y, robot_rz + rotation_direction * increment]

        if state['rotation_count'] > 36: # full rotation without detection (10 deg steps assumed)
            state['rotation_count'] = 0 # reset the rotation counter
            state['search_mode'] = 'far' if state['search_mode'] == 'close' else 'close' #swap search mode
            debug_print("Search mode switched to:", state['search_mode']) #print for debugging

        # Implement search pattern for when marker not detected
        robot_x, robot_y, robot_rz = move_omni.act_pose # rotate to search for aruco
        delta_rz = np.deg2rad(10)  # 10-degree rotation per step
        return [robot_x, robot_y, robot_rz + delta_rz]

print("\n--- Iniciando simulación ---\n")

while True:
    img = update_camera()  # <--- Assign the return value of update_camera() to img
    position_omni = move_omni.act_pose[:2]
    theta_omni = move_omni.act_pose[2]

    if next_pose is None:
        next_pose = get_next_pose(img)
        if next_pose is not None:
            move_omni.set_target_pose(next_pose)

    update_mov()
    
    debug_print("\nIteration:", time.time())
    debug_print("Current Pose:", position_omni, np.rad2deg(theta_omni))
    debug_print("Next Pose:", next_pose)
    debug_print("State:", state)
    debug_print("Detected Markers:", state['detected_markers'])
    debug_print("\n")
    
    p.stepSimulation()