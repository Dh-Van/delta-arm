import sympy as sp
from utils import calculate_end_effector_position, calculate_constraint

def calculate_control_jacobian(distal_angles_numeric, proximal_angles_numeric):
    distal_angles = sp.Matrix(sp.symbols('d1:7'))
    proximal_angles = sp.Matrix(sp.symbols('p1:4'))

    end_effector_pos = calculate_end_effector_position(distal_angles, proximal_angles)
    constraint_eqs = calculate_constraint(distal_angles, proximal_angles)

    jac_end_distal = end_effector_pos.jacobian(distal_angles)
    jac_end_proximal = end_effector_pos.jacobian(proximal_angles)
    jac_constraint_distal = constraint_eqs.jacobian(distal_angles)
    jac_constraint_proximal = constraint_eqs.jacobian(proximal_angles)

    solved_term = jac_constraint_distal.LUsolve(jac_constraint_proximal)
    control_jacobian = jac_end_distal @ -solved_term + jac_end_proximal

    control_jacobian_func = sp.lambdify([distal_angles, proximal_angles], control_jacobian, modules="numpy")
    return control_jacobian_func(distal_angles_numeric, proximal_angles_numeric)

def calculate_gravity_compensation(distal_angles_numeric, proximal_angles_numeric):
    # Define symbolic variables
    distal_angles = sp.Matrix(sp.symbols('d1:7'))
    proximal_angles = sp.Matrix(sp.symbols('p1:4'))
    gravity = sp.Matrix([0, 0, 9.81])

    # Compute end-effector position and gravity torque
    end_effector_pos = calculate_end_effector_position(distal_angles, proximal_angles)
    jac_end_distal = end_effector_pos.jacobian(distal_angles)

    gravity_torque = jac_end_distal.T @ gravity

    # Convert to numerical function and evaluate
    gravity_torque_func = sp.lambdify([distal_angles, proximal_angles], gravity_torque, modules="numpy")
    return gravity_torque_func(distal_angles_numeric, proximal_angles_numeric)


distal_angles_numeric = [1.0485, 1e-5, 1.0485, 1e-5, 1.0485, 1e-5]
proximal_angles_numeric = [0.4556, 0.4556, 0.4556]

# print(calculate_control_jacobian(distal_angles_numeric, proximal_angles_numeric))
print(calculate_gravity_compensation(distal_angles_numeric, proximal_angles_numeric))