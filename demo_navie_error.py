from dolfin import *
import numpy as np

set_log_level(LogLevel.ERROR)

M = 200
a = 1.0
gamma = 1.4
iters = 5
taus = [0.01, 0.005, 0.0025]

results = {}

for tau in taus:
    mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)
    P1_rho = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P1_u = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([P1_rho, P1_u]))

    w_n = Function(W)
    w_k = Function(W)
    
    rho_n, u_n = split(w_n)
    rho_k, u_k = split(w_k)
    v = TestFunction(W)
    phi, psi = split(v)

    initial_density = Expression("1.0 + 2.0 * exp(-20.0 * (x[0]*x[0] + x[1]*x[1]))", degree=2)
    rho_init = project(initial_density, W.sub(0).collapse())
    u_init = project(Constant((0.0, 0.0)), W.sub(1).collapse())
    
    assign(w_n.sub(0), rho_init)
    assign(w_n.sub(1), u_init)
    w_k.assign(w_n)
    mu = Constant(0.01)
    dim = mesh.geometry().dim()
    I = Identity(dim)
    # p_k = a * rho_k**gamma
    F_rho = ((rho_k - rho_n) / tau) * phi * dx - rho_k * dot(u_k, grad(phi)) * dx
    
    # Уравнение импульса (Эйлер)
    # F_u = inner((rho_k * u_k - rho_n * u_n) / tau, psi) * dx \
    #       - inner(rho_k * outer(u_k, u_k), grad(psi)) * dx \
    #       - p_k * div(psi) * dx
          
    c = 1.0
    p_k = c**2 * rho_k + (gamma - 1) * rho_k**gamma

    F_rho = ((rho_k - rho_n) / tau) * phi * dx \
            - rho_k * dot(u_k, grad(phi)) * dx

    F_u = inner((rho_k * u_k - rho_n * u_n) / tau, psi) * dx \
          - inner(rho_k * outer(u_k, u_k), grad(psi)) * dx \
          - p_k * div(psi) * dx

    tau_visc = mu * (grad(u_k) + grad(u_k).T) \
               - (2.0 / 3.0) * mu * div(u_k) * I

    F_u += inner(tau_visc, grad(psi)) * dx

    F = F_rho + F_u
    bc = DirichletBC(W.sub(1), Constant((0.0, 0.0)), "on_boundary")

    dw = TrialFunction(W)
    J = derivative(F, w_k, dw)

    tau_errors = []
    
    for k in range(iters):
        delta_w = Function(W)
        solve(J == -F, delta_w, bc)
        
        w_k.vector()[:] += delta_w.vector()[:]
        
        d_rho_vec = delta_w.sub(0, deepcopy=True).vector()
        rho_k_vec = w_k.sub(0, deepcopy=True).vector()
        
        err = norm(d_rho_vec, 'l2') / norm(rho_k_vec, 'l2')
        tau_errors.append(err)
        
    results[tau] = tau_errors

print(f"Сходимость при первом шаге в {iter} итерациях")
print("-" * 55)
header = f"{'Итер':<10}" + "".join([f"tau = {t:<12}" for t in taus])
print(header)
print("-" * 55)

for i in range(iters):
    row = f"{i+1:<10}"
    for tau in taus:
        row += f"{results[tau][i]:.3e}    "
    print(row)