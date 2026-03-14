from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

tau = 0.01
T = 5
iters = 2
a, gamma = 1.0, 1.4

times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

def run_simulation(M):

    mesh = RectangleMesh(Point(-5,-5), Point(5,5), M, M)

    V_rho = FunctionSpace(mesh, "Lagrange", 1)
    V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

    initial_density = Expression(
        "1.0 + 2.0*exp(-20.0*(x[0]*x[0] + x[1]*x[1]))",
        degree=2
    )

    rho_n = project(initial_density, V_rho)
    u_n = project(Constant((0.0,0.0)), V_u)

    rho_k = Function(V_rho)
    u_k = Function(V_u)

    rho_trial, phi = TrialFunction(V_rho), TestFunction(V_rho)
    u_trial, psi = TrialFunction(V_u), TestFunction(V_u)

    F_rho = (rho_trial-rho_n)/tau*phi*dx \
            - rho_trial*dot(u_k, grad(phi))*dx

    bc_u = DirichletBC(V_u, Constant((0,0)), "on_boundary")

    t = 0

    time_hist = []
    courant_hist = []

    while t < T:

        t += tau

        rho_k.assign(rho_n)
        u_k.assign(u_n)

        for k in range(iters):

            solve(lhs(F_rho)==rhs(F_rho), rho_k)

            p_k = a * rho_k**gamma

            F_u = (rho_k*dot(u_trial,psi) - rho_n*dot(u_n,psi))/tau*dx \
                - inner(rho_k*outer(u_trial,u_k), grad(psi))*dx \
                - p_k*div(psi)*dx

            solve(lhs(F_u)==rhs(F_u), u_k, bc_u)

        rho_n.assign(rho_k)
        u_n.assign(u_k)

        u1, u2 = u_n.split()

        u1_vals = u1.vector().get_local()
        u2_vals = u2.vector().get_local()

        max_u1 = np.abs(u1_vals).max()
        max_u2 = np.abs(u2_vals).max()

        C = (max_u1 + max_u2) * tau * 10.0 / M

        time_hist.append(t)
        courant_hist.append(C)

    return time_hist, courant_hist


# ---- запуск для разных сеток ----

Ms = [50, 100, 200]

results = {}

for M in Ms:
    print("Running M =", M)
    t_hist, C_hist = run_simulation(M)
    results[M] = (t_hist, C_hist)


# ---- график Courant ----

plt.figure(figsize=(8,5))

for M in Ms:
    t_hist, C_hist = results[M]
    plt.plot(t_hist, C_hist, label=f"M = {M}")

plt.xlabel("t")
plt.ylabel("Courant number")
plt.title("Time-evolution of the Courant number")
plt.legend()
plt.grid()

plt.savefig("courant_comparison.png")
plt.show()