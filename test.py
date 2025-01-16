import numpy as np
import sympy as sp

ee_pos = [1e-5, 1, 11.35]

l0, l1, l2, l3 = 2, 5, 9, 2
q11, q12, q13 = sp.symbols("q11, q12, q13")

# Define the equations with q12 measured from the horizontal
z1 = l2 * sp.sin(sp.pi - q12) * sp.sin(q13)
z2 = l0 - l3 + l1 * sp.cos(q11) + l2 * sp.cos(sp.pi - q12)
z3 = l1 * sp.sin(q11) + l2 * sp.sin(sp.pi - q12) * sp.cos(q13)

# Set up the system of equations
equations = [
    sp.Eq(z1, ee_pos[0]),
    sp.Eq(z2, ee_pos[1]),
    sp.Eq(z3, ee_pos[2])
]

# Solve the system of equations
solutions = sp.nsolve(
    equations,
    (q11, q12, q13),
    (sp.rad(10).evalf(), sp.rad(10).evalf(), sp.rad(10).evalf()),  # Initial guess
    verify=False
)

print(solutions)