import numpy as np

# Constants
BASE_RADIUS = 0.165
PROXIMAL_LINK_LENGTH = 0.2
DISTAL_LINK_LENGTH = 0.4
WRIST_FLANGE_WIDTH = 0.0562

def calculate_arm_position(joint_angles):
    joint_angles = np.radians(joint_angles)
    return np.array([
        DISTAL_LINK_LENGTH * np.sin(joint_angles[1] * np.sin(joint_angles[2])),
        BASE_RADIUS - WRIST_FLANGE_WIDTH + PROXIMAL_LINK_LENGTH * np.cos(joint_angles[0]) + DISTAL_LINK_LENGTH * np.cos(joint_angles[1]),
        PROXIMAL_LINK_LENGTH * np.sin(joint_angles[0]) + DISTAL_LINK_LENGTH * np.sin(joint_angles[1]) * np.cos(joint_angles[2])
    ])

def rotation_matrix_z(angle):
    angle = np.radians(angle)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

# Input: [[joint 1 arm angles (3)], [joint 2 arm angles (3)], [joint 3 arm angles (3)]]
def calculate_arm_endpoints(arm_angles):
    return np.array([
        rotation_matrix_z(0) @ calculate_arm_position(arm_angles[0]),
        rotation_matrix_z(120) @ calculate_arm_position(arm_angles[1]),
        rotation_matrix_z(240) @ calculate_arm_position(arm_angles[2]),
    ])

# Input: [6 element vector, 3 element vector]
def calculate_constraint_vector(unmeasured_joint_angles, measured_joint_angles):
    joint_angles = [
        [measured_joint_angles[0], unmeasured_joint_angles[0], unmeasured_joint_angles[1]],
        [measured_joint_angles[1], unmeasured_joint_angles[2], unmeasured_joint_angles[3]],
        [measured_joint_angles[2], unmeasured_joint_angles[4], unmeasured_joint_angles[5]],
    ]

    arm_positions = calculate_arm_endpoints(joint_angles)
    return np.array([
        arm_positions[0] - (rotation_matrix_z(120) @ arm_positions[1]),
        arm_positions[1] - (rotation_matrix_z(240) @ arm_positions[2]),
    ])

def calculate_fk_jacobian(unmeasured_joint_angles, measured_joint_angles):
    initial_constraint_vector = calculate_constraint_vector(unmeasured_joint_angles, measured_joint_angles).flatten()
    # Jacobian has rows that correspond to each unmeasured joint angle and columns that correspond to each constraint equation
    jacobian = np.zeros((len(initial_constraint_vector), len(unmeasured_joint_angles)))
    # Delta value for finite difference approximation
    delta = 1e-5

    # Finite Difference Approximation
    for i in range(len(unmeasured_joint_angles)):
        unmeasured_joint_angles_delta = unmeasured_joint_angles.copy()
        unmeasured_joint_angles_delta[i] += delta

        constraint_vector_delta = calculate_constraint_vector(unmeasured_joint_angles_delta, measured_joint_angles).flatten()

        jacobian[:, i] = (constraint_vector_delta - initial_constraint_vector) / delta

    # Regularization works to ensure that the jacobian is non singular
    return jacobian + np.eye(jacobian.shape[0]) * 1e-6

def calculate_unmeasured_joint_angles(measured_joint_angles, tolerance = 1e-6, max_iterations=200):
    unmeasured_joint_angles_guess = np.zeros(6)
    
    # Newtons method
    for iteration in range(max_iterations):
        constraint_vector = calculate_constraint_vector(unmeasured_joint_angles_guess, measured_joint_angles).flatten()
        jacobian = calculate_fk_jacobian(unmeasured_joint_angles_guess, measured_joint_angles)

        # If the constraint vector is close enough to zero it will return the unmeasured joint angles
        if(np.linalg.norm(constraint_vector) < tolerance):
            print("in here")
            break
        
        # TODO Figure out why this works
        unmeasured_joint_angles_guess += np.linalg.solve(jacobian, -constraint_vector)

    return unmeasured_joint_angles_guess

def calculate_end_effector_position(measured_joint_angles):
    unmeasured_joint_angles = calculate_unmeasured_joint_angles(measured_joint_angles)
    x, y, z = 0, 0, 0
    
    for i in range(3):
        arm_position = calculate_arm_endpoints(unmeasured_joint_angles[i*2:i*2+2], measured_joint_angles[i])
        x += arm_position[0]
        y += arm_position[1]
        z += arm_position[2]

    return np.array([x, y, z]) / 3


measured_joint_angles = [30, 150, 270]  # Example joint angles for each of the 3 proximal arms