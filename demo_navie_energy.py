from dolfin import *
import matplotlib.pyplot as plt
import time

# Опционально: включение оптимизации для ускорения расчетов
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

T = 5
iters = 2
gamma = 1.4

M = 50
taus = [0.01, 0.005, 0.0025]
colors = {0.01: "#1f77b4", 0.005: "#ff7f0e", 0.0025: "#2ca02c"} # Цвета как на вашем графике

def run_energy_solver(M, tau):
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

    time_hist = []
    energy_hist = []

    t = 0.0
    
    # Функция для вычисления полной механической энергии E(t_n)
    # def compute_E(rho_func, u_func):
    #     # Интеграл: 0.5 * rho * |u|^2 + Pi(rho)
    #     # Pi(rho) берем как (rho^gamma) / (gamma - 1)
    #     Pi_rho = (rho_func**gamma) / (gamma - 1.0)
    #     E_form = (0.5 * rho_func * inner(u_func, u_func) + Pi_rho) * dx
    #     return assemble(E_form)
# Функция для вычисления полной механической энергии E(t_n)
    def compute_E(rho_func, u_func):
        # 1. Кинетическая энергия: 0.5 * rho * |u|^2
        E_kin = 0.5 * rho_func * inner(u_func, u_func)
        
        # 2. Потенциальная энергия Pi(rho), соответствующая твоему p_k:
        # Для слагаемого c^2 * rho энергия будет c^2 * ln(rho) * rho (или c^2 * rho для упрощения)
        # Для слагаемого (gamma - 1) * rho^gamma энергия будет rho^gamma
        
        c = 1.0
        # Интегрирование p(rho)/rho^2 дает формулу внутренней энергии:
        Pi_rho = c**2 * rho_func * ln(rho_func) + (rho_func**gamma)
        
        E_form = (E_kin + Pi_rho) * dx
        return assemble(E_form)
    
    time_hist.append(t)
    energy_hist.append(compute_E(rho_n, u_n))

    while t < T:
        t += tau
        rho_k.assign(rho_n)
        u_k.assign(u_n)

        for k in range(iters):
            solve(lhs(F_rho) == rhs(F_rho), rho_k)

            c = 1.0
            p_k = c**2 * rho_k + (gamma - 1)*rho_k**gamma

            F_u = (rho_k*dot(u_trial, psi) - rho_n*dot(u_n, psi))/tau*dx \
                  - inner(rho_k*outer(u_trial, u_k), grad(psi))*dx \
                  - p_k*div(psi)*dx
                  
            # tau_visc = mu * (grad(u_trial) + grad(u_trial).T) - (2.0/3.0) * mu * div(u_trial) * I
            # F_u += inner(tau_visc, grad(psi)) * dx

            solve(lhs(F_u) == rhs(F_u), u_k, bc_u)

        rho_n.assign(rho_k)
        u_n.assign(u_k)

        # Вычисляем энергию на текущем шаге
        time_hist.append(t)
        energy_hist.append(compute_E(rho_n, u_n))

    return time_hist, energy_hist

start_time = time.time()

plt.figure(figsize=(10, 6))

print(f"Начинаю расчет эволюции энергии для M={M}...")
for tau in taus:
    print(f"Считаю для tau = {tau}...")
    t_hist, E_hist = run_energy_solver(M, tau)
    plt.plot(t_hist, E_hist, label=rf"$\tau = {tau}$", color=colors[tau], linewidth=1.5)

plt.xlabel("$t$")
plt.ylabel("$E$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Эволюция_энергии_Эйлера_M50.png", dpi=300)

end_time = time.time()
print(f"Общее время выполнения: {end_time - start_time:.2f} сек.")