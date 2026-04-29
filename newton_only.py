from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time

T = 5.0
tau = 0.005
M = 200
gamma = 1.4
c = 1.0
num_viz_steps = 5
viz_interval = T / (num_viz_steps - 1)
results = []
start_time = time.time()

mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([P1, V1]))

v = TestFunction(W)
w = Function(W)
phi, psi = split(v)
rho, u = split(w)

w_n = Function(W)
rho_n, u_n = split(w_n)

initial_density = Expression("1.0 + 2.0*exp(-2*(x[0]*x[0] + x[1]*x[1]))", degree=2)
initial_velocity = Constant((0.0, 0.0))

assign(w_n.sub(0), project(initial_density, W.sub(0).collapse()))
assign(w_n.sub(1), project(initial_velocity, W.sub(1).collapse()))

results.append((0.0, project(w_n.sub(0), W.sub(0).collapse())))

w.assign(w_n)
bc = DirichletBC(W.sub(1), Constant((0, 0)), "on_boundary")
p = c**2 * rho + (gamma - 1) * rho**gamma

F_rho = (rho - rho_n)/tau * phi * dx - rho * dot(u, grad(phi)) * dx
F_u = dot(rho*u - rho_n*u_n, psi)/tau * dx - inner(rho * outer(u, u), grad(psi)) * dx - p * div(psi) * dx

F = F_rho + F_u
J = derivative(F, w)

problem = NonlinearVariationalProblem(F, w, bc, J)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters['newton_solver']
prm['linear_solver'] = 'mumps'
prm['absolute_tolerance'] = 1E-8
prm['relative_tolerance'] = 1E-7

times = [0.0, 1.0, 2.0, 4.0, 5.0]
results = [] 

t = 0.0
step_count = 0

rho_0 = project(w_n.sub(0), W.sub(0).collapse())
results.append((0.0, rho_0))

while t < T:
    t += tau
    step_count += 1
    solver.solve()
    w_n.assign(w)

    for tt in times:
        if abs(t - tt) < tau / 2:
            rho_current = project(w.sub(0), W.sub(0).collapse())
            results.append((t, rho_current))
            print(f"t = {t:.2f}")

fig, axes = plt.subplots(1, len(results), figsize=(25, 5))
for i, (t_val, rho_val) in enumerate(results):
    plt.sca(axes[i])
    p_plot = plot(rho_val, mode="contourf", cmap="viridis")
    plt.colorbar(p_plot, ax=axes[i])
    axes[i].set_title(f"t={t_val:.2f}")

plt.tight_layout()
plt.savefig("Плотность_Ньютон_Эйлер_200.png")

end_time = time.time()
print(f"Время: {end_time - start_time:.2f}")