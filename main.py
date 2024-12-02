import sympy as sp  

l_0, l_1, l_2, l_3, l_4 = 0.165, 0.2, 0.4, 0.2, 0.0562
m_p, m_d, m_ee = 0.2, 0.1, 0.2
com_p, com_d, com_ee = l_1 / 2, l_2 / 2, l_3
gear_ratio = 1/2

q11, q21, q31 = sp.symbols("q11" "q21" "q31")
q12, q13, q22, q23, q32, q33 = sp.symbols("q12" "q13" "q22" "q23" "q32" "q33")

B = gear_ratio * sp.eye(3)
x = sp.Matrix([q12, q13, q22, q23, q32, q33])
y = sp.Matrix([q11, q21, q31])

def psi(q):
    q11, q12, q13 = q[0], q[1], q[2]
    return sp.Matrix([
        l_2 * sp.sin(q12) * sp.sin(q13),
        l_0 - l_4 + l_1 * sp.cos(q11) + l_2 * sp.cos(q12),
        l_1 * sp.sin(q11) + l_2 * sp.sin(q12) * sp.cos(q13)
    ])

def psi(x, y):
    return psi(sp.Matrix(
        [
            [y[0], x[0], x[1]],
            [y[1], x[2], x[3]],
            [y[2], x[4], x[5]]
        ]
    ))

def h(q):
    q1, q2, q3 = [q[1], q[2], q[3]]
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

def h(x, y):
    return h(sp.Matrix(
        [
            [y[0], x[0], x[1]],
            [y[1], x[2], x[3]],
            [y[2], x[4], x[5]]
        ]
    ))

def H(q):
    return h(q).jacobian(x)

def calculate_x(y, guess, tolerance=1e-5, max_iterations=200):
    x_guess = sp.Matrix(guess)

    for _ in range(max_iterations):
        h_guess = h(x_guess, y)
        J_h = h(x_guess, y).jacobian(x)

        if(all(sp.Abs(h) < tolerance for h in h_guess)):
            break

        delta_x = -J_h.inv() * h_guess
        x_guess += delta_x
    
    return x_guess

def calculate_control_jacobian(x, y):
    return psi(x, y).jacobian(x) * (-(h(x, y).jacobian(x)).inv() * h(x, y).jacobian(y)) + psi(x, y).jacobian(y)

def G(q):
    G = sp.zeros(3, 1)
    for i in range(len(q)):
        q_i = q[i]
        tau_p = com_p * m_p * 9.8 * sp.sin(sp.pi / 2 - q_i[0])
        tau_d = com_d * m_d * 9.8 * sp.sin(sp.pi / 2 - q_i[1])
        # divided by 3 because it should be distributed evenly across all 3 arms
        tau_e = (com_ee * m_ee * 9.8) / 3
        G[i] = tau_p + tau_d + tau_e
    return G

def tau_u(q, tolerance=1e-5, max_iterations=200):
    tau_u_guess = sp.Matrix([0, 0, 0])
    lambda_guess = sp.Matrix([0, 0, 0])
    guess = sp.Matrix.vstack(tau_u_guess, lambda_guess)

    for _ in range(max_iterations):
        F = B * tau_u_guess + H(q).transpose() * lambda_guess - G(q)
        J_F = F.jacobian(guess)

        F_val = F.evalf()
        if all(abs(f) < tolerance for f in F_val):
            break

        delta_guess = -J_F.inv() * F

        guess += delta_guess

    return guess[:3]
#poop
  
