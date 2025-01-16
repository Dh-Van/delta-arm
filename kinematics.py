import sympy as sp
import utils

def forward_kinematics(proximal_angles, initial_guess, tolerance=1e-5, max_iterations=200):
    distal_angle_symbols = sp.symbols('q12,q13,q22,q23,q32,q33') 
    distal_angles_guess = sp.Matrix(distal_angle_symbols)

    numerical_guess = sp.Matrix(initial_guess)

    for _ in range(max_iterations):
        constraint_values = utils.calculate_constraint(distal_angles_guess, proximal_angles)
        constraint_jacobian = constraint_values.jacobian(distal_angles_guess)

        constraint_values = constraint_values.subs(dict(zip(distal_angle_symbols, numerical_guess))).evalf()
        constraint_jacobian = constraint_jacobian.subs(dict(zip(distal_angle_symbols, numerical_guess))).evalf()

        if constraint_values.norm() < tolerance:
            break

        delta_angle = -constraint_jacobian.inv() * constraint_values
        numerical_guess += delta_angle

    numerical_guess = numerical_guess.applyfunc(lambda x: 0 if abs(x) < 1e-5 else x)
    return numerical_guess


def inverse_kinematics_single(end_effector_position):
    servo_angle, link_elevation, link_tilt = sp.symbols("servo_angle link_elevation link_tilt")
    
    z1 = utils.DISTAL_LENGTH * sp.sin(sp.pi - link_elevation) * sp.sin(link_tilt)
    z2 = utils.BASE_RADIUS - utils.END_EFFECTOR_WIDTH + utils.PROXIMAL_LENGTH * sp.cos(servo_angle) + utils.DISTAL_LENGTH * sp.cos(sp.pi - link_elevation)
    z3 = utils.PROXIMAL_LENGTH * sp.sin(servo_angle) + utils.DISTAL_LENGTH * sp.sin(sp.pi - link_elevation) * sp.cos(link_tilt)

    equations = [
        sp.Eq(z1, end_effector_position[0]),
        sp.Eq(z2, end_effector_position[1]),
        sp.Eq(z3, end_effector_position[2])
    ]

    initial_guess = (sp.rad(10).evalf(), sp.rad(10).evalf(), sp.rad(10).evalf())
    solutions = sp.nsolve(equations, (servo_angle, link_elevation, link_tilt), initial_guess, verify=False)

    solutions = solutions.applyfunc(lambda x: 0 if abs(x) < 1e-5 else x)
    return solutions

def inverse_kinematics(end_effector_position):
    arm_1 = inverse_kinematics_single(end_effector_position)
    arm_2 = inverse_kinematics_single(utils.R_2 * sp.Matrix(end_effector_position))
    arm_3 = inverse_kinematics_single(utils.R_3 * sp.Matrix(end_effector_position))

    proximal_angles = [arm_1[0], arm_2[0], arm_3[0]]
    distal_angles = [arm_1[1], arm_1[2], arm_2[1], arm_2[2], arm_3[1], arm_3[2]]

    proximal_angles = [0 if abs(angle) < 1e-5 else angle for angle in proximal_angles]
    distal_angles = [0 if abs(angle) < 1e-5 else angle for angle in distal_angles]

    return proximal_angles, distal_angles

print(inverse_kinematics([0, 0, 10]))