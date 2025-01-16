import sympy as sp

BASE_RADIUS, PROXIMAL_LENGTH, DISTAL_LENGTH, END_EFFECTOR_WIDTH = 2, 5, 9, 2

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

def calculate_end_effector_position(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple, sp.Matrix)):
        arm_angles = args[0]
        return sp.Matrix([
            DISTAL_LENGTH * sp.sin(sp.pi - arm_angles[1]) * sp.sin(arm_angles[2]),
            BASE_RADIUS - END_EFFECTOR_WIDTH + PROXIMAL_LENGTH * sp.cos(arm_angles[0]) + DISTAL_LENGTH * sp.cos(sp.pi - arm_angles[1]),
            PROXIMAL_LENGTH * sp.sin(arm_angles[0]) + DISTAL_LENGTH * sp.sin(sp.pi - arm_angles[1]) * sp.cos(arm_angles[2])
        ]).evalf()
    elif len(args) == 2 and isinstance(args[0], (list, tuple, sp.Matrix)) and isinstance(args[1], (list, tuple, sp.Matrix)):
        x, y = args
        return calculate_end_effector_position(sp.Matrix(
            [
                [y[0], x[0], x[1]],
                [y[1], x[2], x[3]],
                [y[2], x[4], x[5]]
            ]
        ))

def calculate_constraint(distal_angles, proximal_angles):
    joint_angles = sp.Matrix([
        [proximal_angles[0], distal_angles[0], -distal_angles[1]],
        [proximal_angles[1], distal_angles[2], -distal_angles[3]],
        [proximal_angles[2], distal_angles[4], -distal_angles[5]]
    ])
    return sp.Matrix([
        calculate_end_effector_position(joint_angles.row(0)) - R_2 * calculate_end_effector_position(joint_angles.row(1)),
        calculate_end_effector_position(joint_angles.row(0)) - R_3 * calculate_end_effector_position(joint_angles.row(2))
    ])

