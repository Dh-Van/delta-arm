import sympy as sp

base_radius, proximal_length, distal_length, end_effector_radius, offset_distance = 0.165, 0.2, 0.4, 0.2, 0.0562

def calculate_end_effector_position(arm_angles):
    return sp.Matrix([
        distal_length * sp.sin(arm_angles[1]) * sp.sin(arm_angles[2]),
        base_radius - offset_distance + proximal_length * sp.cos(arm_angles[0]) + distal_length * sp.cos(arm_angles[1]),
        proximal_length * sp.sin(arm_angles[0]) + distal_length * sp.sin(arm_angles[1]) * sp.cos(arm_angles[2])
    ])

def calculate_constraint(joint_angles):
    R_2 = sp.Matrix([
        [sp.cos(sp.rad(120)), -sp.sin(sp.rad(120)), 0],
        [sp.sin(sp.rad(120)), sp.cos(sp.rad(120)), 0],
        [0, 0, 1]
    ])

    R_3 = sp.Matrix([
        [sp.cos(sp.rad(240)), -sp.sin(sp.rad(240)), 0],
        [sp.sin(sp.rad(240)), sp.cos(sp.rad(240)), 0],
        [0, 0, 1]
    ])

    return sp.Matrix([
        calculate_end_effector_position(joint_angles[0]) - R_2 * calculate_end_effector_position(joint_angles[1]),
        calculate_end_effector_position(joint_angles[0]) - R_3 * calculate_end_effector_position(joint_angles[2])
    ])

def calculate_constraint(distal_angles, proximal_angles):
    return calculate_constraint(sp.Matrix(
        [
            [proximal_angles[0], distal_angles[0], distal_angles[1]],
            [proximal_angles[1], distal_angles[2], distal_angles[3]],
            [proximal_angles[2], distal_angles[4], distal_angles[5]]
        ]
    ))

def calculate_forward_kinematics(proximal_angles, initial_guess, tolerance=1e-5, max_iterations=200):
    distal_angles_guess = sp.Matrix(initial_guess)

    for _ in range(max_iterations):
        constraint_values = calculate_constraint(distal_angles_guess, proximal_angles)
        constraint_jacobian = constraint_values.jacobian(distal_angles_guess)

        if(all(sp.Abs(constraint_value) < tolerance for constraint_value in constraint_values)):
            break

        delta_angle = -constraint_jacobian.inv() * constraint_values
        distal_angles_guess += delta_angle

    return distal_angles_guess

def calculate_inverse_kinematics_single(end_effector_position):
    distal_twist_angle_s = sp.symbols("distal_twist_angle_s")
    eq = sp.Eq(sp.sqrt(proximal_length**2 - (end_effector_position[2] - end_effector_position[0] * sp.cot(distal_twist_angle_s))**2) \
         - sp.sqrt(distal_length**2 - end_effector_position[0]**2 * sp.csc(distal_twist_angle_s)**2) \
         + base_radius - end_effector_radius - end_effector_position[1], 0)
    
    initial_guess = sp.asin(end_effector_position[0] / distal_length)

    print(type(eq))
    distal_twist_angle = sp.nsolve(eq, distal_twist_angle_s, initial_guess)

    distal_angle = sp.pi - sp.asin(end_effector_position[0] / (distal_length * sp.sin(distal_twist_angle)))
    proximal_angle = sp.asin((end_effector_position[2] - distal_length * sp.sin(distal_angle) * sp.cos(distal_twist_angle)) / proximal_length)

    return [proximal_angle, distal_angle, distal_twist_angle]

def calculate_inverse_kinematics(end_effector_position):
    arm_1 = calculate_inverse_kinematics_single(end_effector_position)

    R_2 = sp.Matrix([
        [sp.cos(sp.rad(120)), -sp.sin(sp.rad(120)), 0],
        [sp.sin(sp.rad(120)), sp.cos(sp.rad(120)), 0],
        [0, 0, 1]
    ])

    rotated_position_2 = R_2 * sp.Matrix(end_effector_position)
    arm_2 = calculate_inverse_kinematics_single(rotated_position_2)

    R_3 = sp.Matrix([
        [sp.cos(sp.rad(240)), -sp.sin(sp.rad(240)), 0],
        [sp.sin(sp.rad(240)), sp.cos(sp.rad(240)), 0],
        [0, 0, 1]
    ])

    rotated_position_3 = R_3 * sp.Matrix(end_effector_position)
    arm_3 = calculate_inverse_kinematics_single(rotated_position_3)

    proximal_angles = [arm_1[0], arm_2[0], arm_3[0]]
    distal_angles = [arm_1[1], arm_1[2], arm_2[1], arm_2[2], arm_3[1], arm_3[2]]

    return proximal_angles, distal_angles



sample_end_effector_pos = [0.0, 1.0, 0.0]
print(calculate_inverse_kinematics(sample_end_effector_pos))