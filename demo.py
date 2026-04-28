from dolfin import *
import matplotlib.pyplot as plt

M = 200
tau = 0.005
T = 5
iters = 2
a, gamma = 1.0, 1.4

times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
res = [] 

mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)
V_rho = FunctionSpace(mesh, "Lagrange", 1)
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

initial_density = Expression("1.0 + 2.0 * exp(-20.0 * (x[0]*x[0] + x[1]*x[1]))", degree=2)
rho_n = project(initial_density, V_rho)
u_n = project(Constant((0.0, 0.0)), V_u)

res.append((0.0, rho_n.copy(deepcopy=True)))

rho_k = Function(V_rho)
u_k = Function(V_u)
rho_trial, phi = TrialFunction(V_rho), TestFunction(V_rho)
u_trial, psi = TrialFunction(V_u), TestFunction(V_u)

F_rho = (rho_trial - rho_n)/tau * phi * dx - rho_trial * dot(u_k, grad(phi)) * dx

bc_u = DirichletBC(V_u, Constant((0, 0)), "on_boundary")

time_hist = []
rho_center = []
rho_min = []
rho_max = []
t = 0.0
while t < T:
    t += tau
    rho_k.assign(rho_n)
    u_k.assign(u_n)
    
    for k in range(iters):
        solve(lhs(F_rho) == rhs(F_rho), rho_k)
        
        p_k = a * rho_k**gamma
        F_u = (rho_k * dot(u_trial, psi) - rho_n * dot(u_n, psi))/tau * dx - inner(rho_k * outer(u_trial, u_k), grad(psi)) * dx - p_k * div(psi) * dx
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

fig, axes = plt.subplots(1, len(res), figsize=(20, 5))

for i, (t, rho_val) in enumerate(res):
    plt.sca(axes[i])
    p = plot(rho_val, mode='contourf')
    plt.colorbar(p, ax=axes[i])
    axes[i].set_title(f"t = {t:.2f}")

plt.tight_layout()
plt.savefig('fig_ideal.png')
plt.show()

plt.figure(figsize=(8,5))

plt.plot(time_hist, rho_center, label="Плотность в центре")
plt.plot(time_hist, rho_min, label="Мин. плотность")
plt.plot(time_hist, rho_max, label="Макс. плотность")

plt.xlabel("t")
plt.ylabel("Плотность")
plt.legend()
plt.grid()
plt.savefig('fig_2.png')
plt.show()