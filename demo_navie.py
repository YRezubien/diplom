from dolfin import *
import matplotlib.pyplot as plt

set_log_level(LogLevel.ERROR)
M = 200
tau = 0.005
T = 5
iters = 2

a = 1.0
gamma = 1.4
nu = 0.01   # коэффициент вязкости

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

rho_prev = Function(V_rho)
u_prev = Function(V_u)

rho_trial = TrialFunction(V_rho)
phi = TestFunction(V_rho)

u_trial = TrialFunction(V_u)
psi = TestFunction(V_u)

F_rho = (rho_trial - rho_n)/tau * phi * dx \
        - rho_trial * dot(u_k, grad(phi)) * dx

bc_u = DirichletBC(V_u, Constant((0.0, 0.0)), "on_boundary")

time_hist = []
rho_center = []
rho_min = []
rho_max = []

t = 0.0
step = 0
print("\nConvergence table (first time step)")
print("Iter    rho error        u error")
print("------------------------------------")

while t < T:

    t += tau
    step += 1

    rho_k.assign(rho_n)
    u_k.assign(u_n)

    for k in range(iters):
        rho_prev.assign(rho_k)
        u_prev.assign(u_k)
        # уравнение массы
        solve(lhs(F_rho) == rhs(F_rho), rho_k)

        # давление
        p_k = a * rho_k**gamma

        # уравнение импульса (Навье–Стокс)
        F_u = (rho_k * dot(u_trial, psi) - rho_n * dot(u_n, psi))/tau * dx \
              - inner(rho_k * outer(u_trial, u_k), grad(psi)) * dx \
              - p_k * div(psi) * dx \
              + nu * inner(sym(grad(u_trial)), sym(grad(psi))) * dx

        solve(lhs(F_u) == rhs(F_u), u_k, bc_u)

        rho_err = norm(rho_k.vector() - rho_prev.vector(), 'l2') / norm(rho_k.vector(), 'l2')
        u_err = norm(u_k.vector() - u_prev.vector(), 'l2') / norm(u_k.vector(), 'l2')

        if step == 1:
            print(f"{k+1:<8}{rho_err:.6e}     {u_err:.6e}")
    rho_n.assign(rho_k)
    u_n.assign(u_k)

    for i in times:
        if abs(t - i) < tau/2:
            res.append((t, rho_n.copy(deepcopy=True)))
            # print(f"Сохранено t = {t:.2f}")

    values = rho_n.vector().get_local()

    time_hist.append(t)
    # rho_center.append(rho_n(Point(0.0, 0.0)))
    # rho_min.append(values.min())
    # rho_max.append(values.max())


# ===== графики плотности =====

fig, axes = plt.subplots(1, len(res), figsize=(20, 5))

for i, (t, rho_val) in enumerate(res):

    plt.sca(axes[i])
    p = plot(rho_val, mode='contourf')

    plt.colorbar(p, ax=axes[i])
    axes[i].set_title(f"t = {t:.2f}")

plt.tight_layout()
plt.savefig('fig_navie_1.png')
plt.show()


# ===== график эволюции плотности =====

# plt.figure(figsize=(8,5))

# plt.plot(time_hist, rho_center, label="Плотность в центре")
# plt.plot(time_hist, rho_min, label="Мин. плотность")
# plt.plot(time_hist, rho_max, label="Макс. плотность")

# plt.xlabel("t")
# plt.ylabel("Плотность")

# plt.legend()
# plt.grid()

# plt.savefig('fig_2_navie.png')
# plt.show()