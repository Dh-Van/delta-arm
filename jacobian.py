import sympy as sp
import numpy as np
import dill
from utils import calculate_end_effector_position, calculate_constraint

def save_control_jacobian(filename="control_jacobian.pkl"):
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

    control_jacobian_func = sp.lambdify([sp.symbols('d1:7'), sp.symbols('p1:4')], control_jacobian, modules="numpy")

    with open(filename, "wb") as f:
        dill.dump(control_jacobian_func, f)

def calculate_control_jacobian(distal_angles_numeric, proximal_angles_numeric, filename="control_jacobian.pkl"):
    with open(filename, "rb") as f:
        control_jacobian_func = dill.load(f)
    return control_jacobian_func(distal_angles_numeric, proximal_angles_numeric)

def save_gravity_compensation(filename="gravity_compensation.pkl"):
    distal_angles = sp.Matrix(sp.symbols('d1:7'))
    proximal_angles = sp.Matrix(sp.symbols('p1:4'))
    gravity = sp.Matrix([0, 0, 9.81])

    end_effector_pos = calculate_end_effector_position(distal_angles, proximal_angles)
    jac_end_distal = end_effector_pos.jacobian(distal_angles)

    gravity_torque = jac_end_distal.T @ gravity

    gravity_torque_func = sp.lambdify([distal_angles, proximal_angles], gravity_torque, modules="numpy")

    with open(filename, "wb") as f:
        dill.dump(gravity_torque_func, f)

def calculate_gravity_compensation(distal_angles_numeric, proximal_angles_numeric, filename="gravity_compensation.pkl"):
    with open(filename, "rb") as f:
        gravity_torque_func = dill.load(f)
    return gravity_torque_func(distal_angles_numeric, proximal_angles_numeric)


distal_angles_numeric = [1.0485, 1e-5, 1.0485, 1e-5, 1.0485, 1e-5]
proximal_angles_numeric = [0.4556, 0.4556, 0.4556]

SAVE=False
if(SAVE):
    save_control_jacobian()
    save_gravity_compensation()

print(calculate_control_jacobian(distal_angles_numeric, proximal_angles_numeric))
print(calculate_gravity_compensation(distal_angles_numeric, proximal_angles_numeric))