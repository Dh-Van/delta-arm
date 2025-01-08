import sympy as sp

base_radius, proximal_length, distal_length, end_effector_radius, offset_distance = 0.165, 0.2, 0.4, 0.2, 0.0562

def calculate_end_effector_position(q):
    q11, q12, q13 = q[0], q[1], q[2]
    return sp.Matrix([
        distal_length * sp.sin(q12) * sp.sin(q13),
        base_radius - offset_distance + proximal_length * sp.cos(q11) + distal_length * sp.cos(q12),
        proximal_length * sp.sin(q11) + distal_length * sp.sin(q12) * sp.cos(q13)
    ])

def calculate_constraint(joint_angles):
    joint_angles_1, joint_angles_2, joint_angles_3 = [joint_angles[1], joint_angles[2], joint_angles[3]]
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
        psi(q1) - R_2 * psi(q2),
        psi(q1) - R_3 * psi(q3)
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
        constraint_value = 