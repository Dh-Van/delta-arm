import sympy as sp

base_radius, proximal_length, distal_length, end_effector_width = 2, 5, 9, 2

def calculate_end_effector_position(arm_angles):
    return sp.Matrix([
        distal_length * sp.sin(arm_angles[1]) * sp.sin(arm_angles[2]),
        base_radius - end_effector_width + proximal_length * sp.cos(arm_angles[0]) + distal_length * sp.cos(arm_angles[1]),
        proximal_length * sp.sin(arm_angles[0]) + distal_length * sp.sin(arm_angles[1]) * sp.cos(arm_angles[2])
    ]).evalf()

def c_ee(x, y):
    return calculate_end_effector_position(sp.Matrix(
        [
            [y[0], x[0], x[1]],
            [y[1], x[2], x[3]],
            [y[2], x[4], x[5]]
        ]
    ))

def calculate_constraint(distal_angles, proximal_angles):
    joint_angles = sp.Matrix([
        [proximal_angles[0], distal_angles[0], distal_angles[1]],
        [proximal_angles[1], distal_angles[2], distal_angles[3]],
        [proximal_angles[2], distal_angles[4], distal_angles[5]]
    ])

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
        calculate_end_effector_position(joint_angles.row(0)) - R_2 * calculate_end_effector_position(joint_angles.row(1)),
        calculate_end_effector_position(joint_angles.row(0)) - R_3 * calculate_end_effector_position(joint_angles.row(2))
    ])
    

def calculate_forward_kinematics(proximal_angles, initial_guess, tolerance=1e-5, max_iterations=200):
    distal_angle_symbols = sp.symbols('q12,q13,q22,q23,q32,q33') 
    distal_angles_guess = sp.Matrix(distal_angle_symbols)

    numerical_guess = sp.Matrix(initial_guess)

    for _ in range(max_iterations):
        constraint_values = calculate_constraint(distal_angles_guess, proximal_angles)
        constraint_jacobian = constraint_values.jacobian(distal_angles_guess)

        constraint_values = constraint_values.subs(dict(zip(distal_angle_symbols, numerical_guess))).evalf()
        constraint_jacobian = constraint_jacobian.subs(dict(zip(distal_angle_symbols, numerical_guess))).evalf()

        if constraint_values.norm() < tolerance:
            break

        delta_angle = -constraint_jacobian.inv() * constraint_values
        numerical_guess += delta_angle

    numerical_guess[0] = (sp.pi - numerical_guess[0]).evalf()
    numerical_guess[2] = (sp.pi - numerical_guess[2]).evalf()
    numerical_guess[4] = (sp.pi - numerical_guess[4]).evalf()

    return numerical_guess


def calculate_inverse_kinematics_single(end_effector_position):
    servo_angle, link_elevation, link_tilt = sp.symbols("servo_angle link_elevation link_tilt")
    
    z1 = distal_length * sp.sin(sp.pi - link_elevation) * sp.sin(link_tilt)
    z2 = base_radius - end_effector_width + proximal_length * sp.cos(servo_angle) + distal_length * sp.cos(sp.pi - link_elevation)
    z3 = proximal_length * sp.sin(servo_angle) + distal_length * sp.sin(sp.pi - link_elevation) * sp.cos(link_tilt)

    equations = [
        sp.Eq(z1, end_effector_position[0]),
        sp.Eq(z2, end_effector_position[1]),
        sp.Eq(z3, end_effector_position[2])
    ]

    initial_guess = (0.1, 0.1, 0.1)
    solutions = sp.nsolve(equations, (servo_angle, link_elevation, link_tilt), initial_guess, verify=False)

    return solutions

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