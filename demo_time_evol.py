from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

M = 50
T = 5
iters = 2
a, gamma = 1.0, 1.4

taus = [0.01, 0.005, 0.0025]
styles = {0.01: ":", 0.005: "--", 0.0025: "-"}

times_to_plot = [1,2,3]

mesh = RectangleMesh(Point(-5,-5), Point(5,5), M, M)

V_rho = FunctionSpace(mesh, "Lagrange", 1)
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

x_line = np.linspace(0,5,25)

plt.figure(figsize=(8,5))

for tau in taus:

    initial_density = Expression(
        "1.0 + 2.0*exp(-20*(x[0]*x[0] + x[1]*x[1]))",
        degree=2)

    rho_n = project(initial_density, V_rho)
    u_n = project(Constant((0,0)), V_u)

    rho_k = Function(V_rho)
    u_k = Function(V_u)

    rho_trial, phi = TrialFunction(V_rho), TestFunction(V_rho)
    u_trial, psi = TrialFunction(V_u), TestFunction(V_u)

    F_rho = (rho_trial - rho_n)/tau * phi*dx \
            - rho_trial*dot(u_k, grad(phi))*dx

    bc_u = DirichletBC(V_u, Constant((0,0)), "on_boundary")

    saved_profiles = {}

    t = 0

    while t < T:

        t += tau

        rho_k.assign(rho_n)
        u_k.assign(u_n)

        for k in range(iters):

            solve(lhs(F_rho)==rhs(F_rho), rho_k)

            p_k = a*rho_k**gamma

            F_u = (rho_k*dot(u_trial,psi) - rho_n*dot(u_n,psi))/tau*dx \
                  - inner(rho_k*outer(u_trial,u_k), grad(psi))*dx \
                  - p_k*div(psi)*dx

            solve(lhs(F_u)==rhs(F_u), u_k, bc_u)

        rho_n.assign(rho_k)
        u_n.assign(u_k)

        for tt in times_to_plot:
            if abs(t-tt) < tau/2:
                saved_profiles[tt] = rho_n.copy(deepcopy=True)

    for tt in times_to_plot:

        rho_vals = [saved_profiles[tt](x,0) for x in x_line]

        plt.plot(
            x_line,
            rho_vals,
            linestyle=styles[tau]
        )

plt.grid()
plt.legend()

plt.savefig("figure4.png")
plt.show()

end_time = time.time()

print("Время выполнения:", end_time - start_time, "с.")