from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time

T = 5
iters = 2
gamma = 1.4

Ms = [50, 100, 200]
times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
taus = [0.01, 0.005, 0.0025]
styles = {0.01: ":", 0.005: "--", 0.0025: "-"}

start_time = time.time()

def run_solver(M, tau, save_times=None, compute_courant=False):
    t = 0.0
    mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)

    V_rho = FunctionSpace(mesh, "Lagrange", 1)
    V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

    initial_density = Expression("1.0 + 2.0*exp(-20*(x[0]*x[0] + x[1]*x[1]))", degree=2)

    rho_n = project(initial_density, V_rho)
    u_n = project(Constant((0, 0)), V_u)

    rho_k = Function(V_rho)
    u_k = Function(V_u)

    rho_trial, phi = TrialFunction(V_rho), TestFunction(V_rho)
    u_trial, psi = TrialFunction(V_u), TestFunction(V_u)

    mu = Constant(0.01)
    dim = mesh.geometry().dim()
    I = Identity(dim)

    F_rho = (rho_trial - rho_n)/tau*phi*dx - rho_trial*dot(u_k, grad(phi))*dx

    bc_u = DirichletBC(V_u, Constant((0, 0)), "on_boundary")

    saved_profiles = {}
    time_hist = []
    rho_center = []
    rho_min = []
    rho_max = []
    courant_hist = []

    dx_mesh = 10.0 / M
    if save_times is not None and 0.0 in save_times:
        saved_profiles[0.0] = rho_n.copy(deepcopy=True)

    while t < T:
        t += tau
        rho_k.assign(rho_n)
        u_k.assign(u_n)

        for k in range(iters):
            solve(lhs(F_rho) == rhs(F_rho), rho_k)

            # p_k = rho_k**gamma
            c = 1.0
            p_k = c**2 * rho_k + (gamma - 1)*rho_k**gamma

            F_u = (rho_k*dot(u_trial, psi) - rho_n*dot(u_n, psi))/tau*dx - inner(rho_k*outer(u_trial, u_k), grad(psi))*dx - p_k*div(psi)*dx
            tau_visc = mu * (grad(u_trial) + grad(u_trial).T) - (2.0/3.0) * mu * div(u_trial) * I
            F_u += inner(tau_visc, grad(psi)) * dx

            solve(lhs(F_u) == rhs(F_u), u_k, bc_u)

        rho_n.assign(rho_k)
        u_n.assign(u_k)

        values = rho_n.vector().get_local()

        time_hist.append(t)
        rho_center.append(rho_n(Point(0, 0)))
        rho_min.append(values.min())
        rho_max.append(values.max())

        if save_times is not None:
            for tt in save_times:
                if abs(t - tt) < tau / 2:
                    saved_profiles[tt] = rho_n.copy(deepcopy=True)

        if compute_courant:
            u1, u2 = u_n.split()
            u1_vals = u1.vector().get_local()
            u2_vals = u2.vector().get_local()
            vel = np.sqrt(u1_vals**2 + u2_vals**2)
            C = vel.max() * tau / dx_mesh
            courant_hist.append(C)

    return {
        "profiles": saved_profiles,
        "time": time_hist,
        "rho_center": rho_center,
        "rho_min": rho_min,
        "rho_max": rho_max,
        "courant": courant_hist
    }

def density(M, tau):
    data = run_solver(M, tau, save_times=times)
    res = [(t, data["profiles"][t]) for t in times if t in data["profiles"]]
    return res, data["time"], data["rho_center"], data["rho_max"], data["rho_min"]

def time_evol(M, taus):
    x_line = np.linspace(0, 5, 25)
    for tau in taus:
        data = run_solver(M, tau, save_times=times)
        for tt in times:
            if tt not in data["profiles"]:
                continue
            rho = data["profiles"][tt]
            rho_vals = [rho(x, 0) for x in x_line]
            plt.plot(x_line, rho_vals, linestyle=styles[tau])

def courant(M, tau):
    data = run_solver(M, tau, compute_courant=True)
    return data["time"], data["courant"]

print("Начинаю плотность (Навье-Стокса):")
res, time_hist, rho_center, rho_max, rho_min = density(200, 0.005)

fig, axes = plt.subplots(1, len(res), figsize=(20, 5))
for i, (t, rho_val) in enumerate(res):
    plt.sca(axes[i])
    p = plot(rho_val, mode="contourf")
    plt.colorbar(p, ax=axes[i])
    axes[i].set_title(f"t={t:.2f}")
plt.tight_layout()
plt.savefig("Плотность_НавьеСтокса.png")

plt.figure(figsize=(8, 5))
plt.plot(time_hist, rho_center, label="Плотность в центре")
plt.plot(time_hist, rho_min, label="Мин. плотность")
plt.plot(time_hist, rho_max, label="Макс. плотность")
plt.xlabel("t")
plt.ylabel("Плотность")
plt.legend()
plt.grid()
plt.savefig("Эволюция_плотности_НавьеСтокса.png")

print("Начинаю эволюцию по tau:")
plt.figure(figsize=(8, 5))
time_evol(50, taus)
plt.grid()
plt.legend()
plt.savefig("Эволюция_времени_НавьеСтокса.png")

print("Начинаю Куранта:")
results = {}
for M in Ms:
    print("Running M =", M)
    t_hist, C_hist = courant(M, 0.01)
    results[M] = (t_hist, C_hist)

plt.figure(figsize=(8, 5))
for M in Ms:
    t_hist, C_hist = results[M]
    plt.plot(t_hist, C_hist, label=f"M={M}")
plt.xlabel("t")
plt.ylabel("C")
plt.title("Эволюция числа Куранта")
plt.legend()
plt.grid()
plt.savefig("Эволюция_Куранта_НавьеСтокса.png")

end_time = time.time()
print("Время выполнения: ", end_time - start_time)