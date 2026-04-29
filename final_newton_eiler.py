from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time

T = 5
gamma = 1.4
c = 1.0

Ms = [50, 100, 200]
times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
taus = [0.01, 0.005, 0.0025]
styles = {0.01: ":", 0.005: "--", 0.0025: "-"}

start_time = time.time()

def run_solver_newton(M, tau, save_times=None, compute_courant=False):
    t = 0.0
    mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)

    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([P1, V1]))

    w = Function(W)
    w_n = Function(W)
    
    rho, u = split(w)
    rho_n, u_n = split(w_n)
    v_test = TestFunction(W)
    phi, psi = split(v_test)

    initial_density = Expression("1.0 + 2.0*exp(-20*(x[0]*x[0] + x[1]*x[1]))", degree=2)
    assign(w_n.sub(0), project(initial_density, W.sub(0).collapse()))
    assign(w_n.sub(1), project(Constant((0, 0)), W.sub(1).collapse()))
    
    w.assign(w_n)
    bc_u = DirichletBC(W.sub(1), Constant((0, 0)), "on_boundary")
    p = c**2 * rho + (gamma - 1) * rho**gamma
    
    F_rho = (rho - rho_n)/tau * phi * dx - rho * dot(u, grad(phi)) * dx
    F_u = dot(rho*u - rho_n*u_n, psi)/tau * dx \
          - inner(rho * outer(u, u), grad(psi)) * dx \
          - p * div(psi) * dx
    
    F = F_rho + F_u
    J = derivative(F, w)

    problem = NonlinearVariationalProblem(F, w, bc_u, J)
    solver = NonlinearVariationalSolver(problem)
    
    prm = solver.parameters
    prm['newton_solver']['linear_solver'] = 'mumps'
    prm['newton_solver']['relative_tolerance'] = 1e-7
    prm['newton_solver']['maximum_iterations'] = 20

    saved_profiles = {}
    time_hist, rho_center, rho_min, rho_max, courant_hist = [], [], [], [], []
    dx_mesh = 10.0 / M

    if save_times is not None and 0.0 in save_times:
        saved_profiles[0.0] = w_n.sub(0).copy(deepcopy=True)

    while t < T:
        t += tau
        solver.solve()
        w_n.assign(w)

        rho_current = w.sub(0, deepcopy=True)
        u_current = w.sub(1, deepcopy=True)
        values = rho_current.vector().get_local()

        time_hist.append(t)
        rho_center.append(rho_current(Point(0, 0)))
        rho_min.append(values.min())
        rho_max.append(values.max())

        if save_times is not None:
            for tt in save_times:
                if abs(t - tt) < tau / 2:
                    saved_profiles[tt] = rho_current.copy(deepcopy=True)

        if compute_courant:
            u1, u2 = u_current.split(True)
            u1_vals = u1.vector().get_local()
            u2_vals = u2.vector().get_local()
            vel = np.sqrt(u1_vals**2 + u2_vals**2)
            C = vel.max() * tau / dx_mesh
            courant_hist.append(C)

    return {
        "profiles": saved_profiles, "time": time_hist,
        "rho_center": rho_center, "rho_min": rho_min,
        "rho_max": rho_max, "courant": courant_hist
    }

def density_newton(M, tau):
    data = run_solver_newton(M, tau, save_times=times)
    res = [(t, data["profiles"][t]) for t in times if t in data["profiles"]]
    return res, data["time"], data["rho_center"], data["rho_max"], data["rho_min"]

print("Запуск метода Ньютона (Плотность):")
res, time_hist, rho_center, rho_max, rho_min = density_newton(100, 0.005)

fig, axes = plt.subplots(1, len(res), figsize=(20, 5))
for i, (t, rho_val) in enumerate(res):
    plt.sca(axes[i])
    p = plot(rho_val, mode="contourf")
    plt.colorbar(p, ax=axes[i])
    axes[i].set_title(f"t={t:.2f}")
plt.savefig("Плотность_Ньютон.png")

plt.figure(figsize=(8, 5))
plt.plot(time_hist, rho_center, label="Центр (Ньютон)")
plt.plot(time_hist, rho_min, label="Мин")
plt.plot(time_hist, rho_max, label="Макс")
plt.legend()
plt.grid()
plt.savefig("Эволюция_плотности_Ньютон.png")

end_time = time.time()
print("Общее время выполнения метода Ньютона: ", end_time - start_time)