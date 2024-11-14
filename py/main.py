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

## INICIALIZACIÓN DE VARIABLES Y OBJETOS

# DEBUG
DEBUG = True

# Define the sequence of marker IDs to detect
marker_sequence =  [1, 301, 401, 402, 100, 100, 305, 100, 307, 200, 100, 406]

# Initialize the state dictionary
state = {
    'current_marker_index': 0,
    'phase': 'approach'
}

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


## LOOP PRINCIPAL

# Variables con valores iniciales, pueden usar estas y/o crear las suyas propias
# state = 0         # Controla que aruco se debe buscar
substate = 0      # Controla que accion se debe hacer
t_cam = 0         # Controla la frecuencia de la cámara
t_mov = 0         # Controla la frecuencia de movimiento
img = None        # Imagen de la cámara
next_pose = None  # Siguiente pose del robot

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

def detect_aruco(image):
    """
    Detect ArUco markers in the current camera frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use the correct ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    # Adjust detection parameters for better accuracy
    parameters.adaptiveThreshConstant = 7
    parameters.minCornerDistanceRate = 0.05
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.03  # Lower this value to detect smaller markers
    parameters.maxMarkerPerimeterRate = 4.0  # Increase this value to detect larger markers
    parameters.polygonalApproxAccuracyRate = 0.05  # Adjust for better accuracy
    parameters.minOtsuStdDev = 5.0  # Adjust for better thresholding

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
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
        # Keep position, update only rotation
        return [robot_x, robot_y, robot_rz]

def get_next_pose(img):
    """
    Process camera image, update navigation state, and calculate next position.
    """
    global state, marker_sequence
    
    detected_markers = detect_aruco(img)
    current_marker_id = marker_sequence[state['current_marker_index']]
    marker_detected = next((m for m in detected_markers if m['id'] == current_marker_id), None)

    if state['phase'] == 'finished':
        return move_omni.act_pose

    if 'rotation_count' not in state:
        state['rotation_count'] = 0

    if state['phase'] == 'rotate':
        state['rotation_count'] += 1
        if state['rotation_count'] > 36:  # Assuming 10-degree increments
            debug_print("Rotation limit reached, moving forward.")
            state['rotation_count'] = 0
            state['phase'] = 'approach'
            state['current_marker_index'] = min(state['current_marker_index'] + 1, len(marker_sequence) - 1)
            # Move forward slightly
            robot_x, robot_y, robot_rz = move_omni.act_pose
            delta_x = 0.1 * np.cos(robot_rz)
            delta_y = 0.1 * np.sin(robot_rz)
            return [robot_x + delta_x, robot_y + delta_y, robot_rz]

    if marker_detected:
        return calculate_next_position(marker_detected)
    else:
        # No marker detected, implement search pattern
        robot_x, robot_y, robot_rz = move_omni.act_pose
        state['rotation_count'] += 1
        
        if state['rotation_count'] > 36:  # Complete rotation, move forward
            state['rotation_count'] = 0
            delta_x = 0.1 * np.cos(robot_rz)
            delta_y = 0.1 * np.sin(robot_rz)
            return [robot_x + delta_x, robot_y + delta_y, robot_rz]
        else:
            # Continue rotating to search
            return [robot_x, robot_y, robot_rz + np.pi/18]  # 10-degree rotation

while True:
    update_camera()
    position_omni = move_omni.act_pose[:2]
    theta_omni = move_omni.act_pose[2]

    if next_pose is None:
        next_pose = get_next_pose(img)
        if next_pose is not None:
            move_omni.set_target_pose(next_pose)

    update_mov()
    p.stepSimulation()
