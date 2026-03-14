from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# параметры
M = 50
tau = 0.005
T = 3

a = 1.0
gamma = 1.4

K_values = [1, 2, 5]
times = [1, 2, 3]

styles = {
    1: '--',
    2: ':',
    5: '-'
}

mesh = RectangleMesh(Point(-5,-5), Point(5,5), M, M)

V_rho = FunctionSpace(mesh,"Lagrange",1)
V_u = VectorFunctionSpace(mesh,"Lagrange",1)

initial_density = Expression(
    "1.0 + 2.0*exp(-20*(x[0]*x[0] + x[1]*x[1]))",
    degree=2
)

bc_u = DirichletBC(V_u, Constant((0,0)), "on_boundary")

rho_trial = TrialFunction(V_rho)
phi = TestFunction(V_rho)

u_trial = TrialFunction(V_u)
psi = TestFunction(V_u)

x_vals = np.linspace(0,5,400)

plt.figure(figsize=(8,6))

for K in K_values:

    print(f"Simulation for K = {K}")

    rho_n = project(initial_density, V_rho)
    u_n = project(Constant((0.0,0.0)), V_u)

    rho_k = Function(V_rho)
    u_k = Function(V_u)

    saved = {}

    t = 0.0

    while t < T:

        t += tau

        rho_k.assign(rho_n)
        u_k.assign(u_n)

        for k in range(K):

            F_rho = (rho_trial-rho_n)/tau*phi*dx \
                  - rho_trial*dot(u_k,grad(phi))*dx

            solve(lhs(F_rho)==rhs(F_rho), rho_k)

            p_k = a*rho_k**gamma

            F_u = (rho_k*dot(u_trial,psi)-rho_n*dot(u_n,psi))/tau*dx \
                - inner(rho_k*outer(u_trial,u_k),grad(psi))*dx \
                - p_k*div(psi)*dx

            solve(lhs(F_u)==rhs(F_u), u_k, bc_u)

        rho_n.assign(rho_k)
        u_n.assign(u_k)

        for time in times:
            if abs(t-time) < tau/2:
                saved[time] = rho_n.copy(deepcopy=True)

    for time in times:

        rho_slice = [saved[time](Point(x,0)) for x in x_vals]

        plt.plot(
            x_vals,
            rho_slice,
            styles[K],
            linewidth=2,
            label=f"t={time}, K={K}"
        )

plt.xlabel("x")
plt.ylabel("rho")
plt.title("Solution at different times and iterations (M=50)")
plt.grid()
plt.legend()

plt.savefig("solution_all_times.png", dpi=300)
plt.show()