from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()
M = 50
tau = 0.005
T = 5
iters = 2
a, gamma = 1.0, 1.4
 
times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
res = []

mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)
V_rho = FunctionSpace(mesh, "Lagrange", 1)
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

initial_density = Expression(
    "1.0 + 2.0 * exp(-20.0 * (x[0]*x[0] + x[1]*x[1]))",
    degree=2
)

rho_n = project(initial_density, V_rho)
u_n = project(Constant((0.0, 0.0)), V_u)

res.append((0.0, rho_n.copy(deepcopy=True)))

rho_k = Function(V_rho)
u_k = Function(V_u)

rho_trial, phi = TrialFunction(V_rho), TestFunction(V_rho)
u_trial, psi = TrialFunction(V_u), TestFunction(V_u)

mu = Constant(0.01)
dim = mesh.geometry().dim()
I = Identity(dim)

F_rho = (rho_trial - rho_n)/tau * phi*dx - rho_trial*dot(u_k, grad(phi))*dx
bc_u = DirichletBC(V_u, Constant((0, 0)), "on_boundary")

time_hist = []
rho_center = []
rho_min = []
rho_max = []
courant_hist = []

t = 0.0

while t < T:

    t += tau

    rho_k.assign(rho_n)
    u_k.assign(u_n)

    for k in range(iters):

        # solve(lhs(F_rho) == rhs(F_rho), rho_k)

        # p_k = a * rho_k**gamma

        # F_u = (rho_k * dot(u_trial, psi) - rho_n * dot(u_n, psi))/tau * dx \
        #       - inner(rho_k * outer(u_trial, u_k), grad(psi)) * dx \
        #       - p_k * div(psi) * dx

        # solve(lhs(F_u) == rhs(F_u), u_k, bc_u)
        c = 1.0
        p_k = c**2 * rho_k + (gamma - 1)*rho_k**gamma

        F_u = (rho_k*dot(u_trial, psi) - rho_n*dot(u_n, psi))/tau*dx \
                - inner(rho_k*outer(u_trial, u_k), grad(psi))*dx \
                - p_k*div(psi)*dx
                
        tau_visc = mu * (grad(u_trial) + grad(u_trial).T) - (2.0/3.0) * mu * div(u_trial) * I
        F_u += inner(tau_visc, grad(psi)) * dx

        solve(lhs(F_u) == rhs(F_u), u_k, bc_u)

    rho_n.assign(rho_k)
    u_n.assign(u_k)

    for i in times:
        if abs(t - i) < tau / 2:
            res.append((t, rho_n.copy(deepcopy=True)))
            print(f"Сохранено t = {t:.2f}")

    values = rho_n.vector().get_local()

    time_hist.append(t)
    rho_center.append(rho_n(Point(0.0, 0.0)))
    rho_min.append(values.min())
    rho_max.append(values.max())

    # ---- Courant number ----

    u1, u2 = u_n.split()

    u1_vals = u1.vector().get_local()
    u2_vals = u2.vector().get_local()

    max_u1 = np.abs(u1_vals).max()
    max_u2 = np.abs(u2_vals).max()

    C = (max_u1 + max_u2) * tau * 10.0 / M
    courant_hist.append(C)

# ---- Density contour plots ----

fig, axes = plt.subplots(1, len(res), figsize=(20, 5))

for i, (t, rho_val) in enumerate(res):

    plt.sca(axes[i])

    p = plot(rho_val, mode='contourf')

    plt.colorbar(p, ax=axes[i])

    axes[i].set_title(f"t = {t:.2f}")

plt.figure(figsize=(8,5))

plt.plot(time_hist, courant_hist)

plt.xlabel("t")
plt.ylabel("Courant number C(t)")
plt.title("Time-evolution of the Courant number")

plt.grid()

plt.savefig("courant.png")
plt.show()
end_time = time.time()

print("Время выполнения:", end_time - start_time, "с.")