from dolfin import *

set_log_level(LogLevel.ERROR)

M = 200
iters = 3
a = 1.0
gamma = 1.4

taus = [0.01, 0.005, 0.0025]

print("\nConvergence table (first time step)")
print("tau        iter1        iter2        iter3")
print("------------------------------------------------")

for tau in taus:

    mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)

    V_rho = FunctionSpace(mesh, "Lagrange", 1)
    V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

    initial_density = Expression(
        "1.0 + 2.0 * exp(-20.0 * (x[0]*x[0] + x[1]*x[1]))",
        degree=2
    )

    rho_n = project(initial_density, V_rho)
    u_n = project(Constant((0.0, 0.0)), V_u)

    rho_k = Function(V_rho)
    u_k = Function(V_u)

    rho_prev = Function(V_rho)
    u_prev = Function(V_u)

    rho_trial = TrialFunction(V_rho)
    phi = TestFunction(V_rho)

    u_trial = TrialFunction(V_u)
    psi = TestFunction(V_u)

    F_rho = (rho_trial - rho_n)/tau * phi * dx - rho_trial * dot(u_k, grad(phi)) * dx

    bc_u = DirichletBC(V_u, Constant((0, 0)), "on_boundary")

    rho_k.assign(rho_n)
    u_k.assign(u_n)

    errors = []

    for k in range(iters):

        rho_prev.assign(rho_k)
        u_prev.assign(u_k)

        solve(lhs(F_rho) == rhs(F_rho), rho_k)

        p_k = a * rho_k**gamma
        F_u = (rho_k * dot(u_trial, psi) - rho_n * dot(u_n, psi))/tau * dx - inner(rho_k * outer(u_trial, u_k), grad(psi)) * dx - p_k * div(psi) * dx
        solve(lhs(F_u) == rhs(F_u), u_k, bc_u)

        rho_err = norm(rho_k.vector() - rho_prev.vector(), 'l2') / norm(rho_k.vector(), 'l2')
        errors.append(rho_err)

    print(f"{tau:<10}{errors[0]:.2e}    {errors[1]:.2e}    {errors[2]:.2e}")