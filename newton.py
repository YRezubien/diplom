# from dolfin import *
# import matplotlib.pyplot as plt
# import numpy as np

# # Параметры задачи
# T = 5
# tau = 0.005
# M = 50
# gamma = 1.4
# c = 1.0
# tol_newton = 1e-20
# max_iter_newton = 10

# # Сетка и пространства функций
# mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)
# V_rho = FunctionSpace(mesh, "Lagrange", 1)
# V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

# # Смешанное пространство (rho, ux, uy)
# P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# element = MixedElement([P1, P1, P1])
# V_mixed = FunctionSpace(mesh, element)

# # Начальные условия
# initial_density = Expression("1.0 + 2.0*exp(-20*(x[0]*x[0] + x[1]*x[1]))", degree=2)
# rho0 = project(initial_density, V_rho)
# u0 = Function(V_u) # По умолчанию (0,0)

# # Граничное условие для скорости
# bc_u = DirichletBC(V_u, Constant((0, 0)), "on_boundary")

# class EulerNewtonProblem:
#     def __init__(self, rho_n, u_n, tau):
#         self.rho_n = rho_n
#         self.u_n = u_n
#         self.tau = tau
#         self.w = Function(V_mixed)
#         self.dw = TrialFunction(V_mixed)
#         self.w_test = TestFunction(V_mixed)
        
#         rho, ux, uy = split(self.w)
#         u = as_vector([ux, uy])
#         phi_rho, psi_ux, psi_uy = split(self.w_test)
#         psi_u = as_vector([psi_ux, psi_uy])

#         p = c**2 * rho + (gamma - 1) * rho**gamma

#         # Уравнение неразрывности
#         self.F_rho = (rho - self.rho_n) / self.tau * phi_rho * dx - rho * dot(u, grad(phi_rho)) * dx
#         # Уравнение импульса
#         self.F_mom = (rho * dot(u, psi_u) - self.rho_n * dot(self.u_n, psi_u)) / self.tau * dx \
#                      - inner(rho * outer(u, u), grad(psi_u)) * dx \
#                      - p * div(psi_u) * dx
        
#         self.F = self.F_rho + self.F_mom
#         self.J = derivative(self.F, self.w, self.dw)

# def solve_newton_step(rho_n, u_n, tau):
#     w = Function(V_mixed)
    
#     # ПРАВИЛЬНОЕ ПРИСВАИВАНИЕ (FunctionAssigner)
#     # Переносим из (V_rho, V_u.sub(0), V_u.sub(1)) в V_mixed
#     assigner = FunctionAssigner(V_mixed, [V_rho, V_u.sub(0), V_u.sub(1)])
#     assigner.assign(w, [rho_n, u_n.sub(0), u_n.sub(1)])

#     problem = EulerNewtonProblem(rho_n, u_n, tau)
#     problem.w.assign(w) 

#     # BC для смешанного пространства (ux=0, uy=0 -> sub(1) и sub(2))
#     bc_mixed = [DirichletBC(V_mixed.sub(1), Constant(0), "on_boundary"),
#                 DirichletBC(V_mixed.sub(2), Constant(0), "on_boundary")]

#     residuals = []
#     for i in range(max_iter_newton):
#         A = assemble(problem.J)
#         b = assemble(problem.F)
        
#         for bc in bc_mixed:
#             bc.apply(A, b)
            
#         du = Function(V_mixed)
#         solve(A, du.vector(), -b)
#         problem.w.vector().axpy(1.0, du.vector())
        
#         res_norm = b.norm("l2")
#         residuals.append(res_norm)
        
#         if res_norm < tol_newton:
#             break
            
#     # Извлечение результатов
#     rho_new = project(problem.w.sub(0), V_rho)
#     # Для скорости создаем вектор из компонент
#     u_new = project(as_vector([problem.w.sub(1), problem.w.sub(2)]), V_u)
    
#     return rho_new, u_new, residuals

# # ---- Метод Вабищевича ----
# def solve_vabishchevich_step(rho_n, u_n, tau, iters=10):
#     rho_k = Function(V_rho)
#     u_k = Function(V_u)
#     rho_k.assign(rho_n)
#     u_k.assign(u_n)

#     changes = []
#     for k in range(iters):
#         rho_old_vec = rho_k.vector().copy()
        
#         # Шаг 1: Плотность
#         rho_trial, phi = TrialFunction(V_rho), TestFunction(V_rho)
#         a_rho = (rho_trial / tau) * phi * dx
#         L_rho = (rho_n / tau) * phi * dx + rho_trial * dot(u_k, grad(phi)) * dx
#         # Заметка: в неявных схемах итерации помогают линеаризовать dot(u, grad(rho))
#         solve(lhs(a_rho - L_rho) == rhs(a_rho - L_rho), rho_k)

#         # Шаг 2: Скорость
#         u_trial, psi = TrialFunction(V_u), TestFunction(V_u)
#         p_k = c**2 * rho_k + (gamma - 1)*rho_k**gamma
#         F_u = (rho_k * dot(u_trial, psi) - rho_n * dot(u_n, psi))/tau * dx \
#               - inner(rho_k * outer(u_trial, u_k), grad(psi)) * dx \
#               - p_k * div(psi) * dx
#         solve(lhs(F_u) == rhs(F_u), u_k, bc_u)

#         diff = rho_k.vector() - rho_old_vec
#         changes.append(diff.norm("l2") / (rho_k.vector().norm("l2") + 1e-12))
        
#     return rho_k, u_k, changes

# # Запуск
# rho_n = rho0
# u_n = u0

# rho_nw, u_nw, res_nw = solve_newton_step(rho_n, u_n, tau)
# rho_vb, u_vb, ch_vb = solve_vabishchevich_step(rho_n, u_n, tau, iters=10)

# # Визуализация
# plt.figure(figsize=(8, 5))
# plt.semilogy(res_nw, 'bo-', label='Newton (Residual)')
# plt.semilogy(ch_vb, 'rs-', label='Vabishchevich (Delta Rho)')
# plt.legend()
# plt.grid(True)
# plt.title("Convergence Comparison")
# plt.show()

# print(f"{'Итерация':<10} | {'Newton (Residual)':<20} | {'Vabishchevich (Delta Rho)':<25}")
# print("-" * 65)

# max_len = max(len(res_nw), len(ch_vb))
# for i in range(max_len):
#     # Извлекаем значения, если они есть для данной итерации
#     newton_val = f"{res_nw[i]:.2e}" if i < len(res_nw) else "-"
#     vabish_val = f"{ch_vb[i]:.2e}" if i < len(ch_vb) else "-"
    
#     print(f"{i:<10} | {newton_val:<20} | {vabish_val:<25}")

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Параметры задачи
tau = 0.005
M = 300
gamma = 1.4
c = 1.0
tol = 1e-20  # Общая точность для обоих методов
max_iter = 10

# Сетка
mesh = RectangleMesh(Point(-5, -5), Point(5, 5), M, M)
V_rho = FunctionSpace(mesh, "Lagrange", 1)
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

# Смешанное пространство для Ньютона
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement([P1, P1, P1])
V_mixed = FunctionSpace(mesh, element)

# Начальные условия (одинаковые для обоих)
initial_density = Expression("1.0 + 2.0*exp(-20*(x[0]*x[0] + x[1]*x[1]))", degree=2)
rho_n = project(initial_density, V_rho)
u_n = Function(V_u)

# --- МЕТОД НЬЮТОНА ---
def solve_newton(rho_n, u_n, tau, tol):
    w = Function(V_mixed)
    # Начальное приближение из предыдущего слоя
    assigner = FunctionAssigner(V_mixed, [V_rho, V_u.sub(0), V_u.sub(1)])
    assigner.assign(w, [rho_n, u_n.sub(0), u_n.sub(1)])
    
    rho, ux, uy = split(w)
    u = as_vector([ux, uy])
    phi_rho, psi_ux, psi_uy = split(TestFunction(V_mixed))
    psi_u = as_vector([psi_ux, psi_uy])
    dw = TrialFunction(V_mixed)

    p = c**2 * rho + (gamma - 1) * rho**gamma
    F = ((rho - rho_n)/tau * phi_rho - rho * dot(u, grad(phi_rho))) * dx + \
        ((rho*dot(u, psi_u) - rho_n*dot(u_n, psi_u))/tau - \
         inner(rho*outer(u, u), grad(psi_u)) - p*div(psi_u)) * dx
    
    J = derivative(F, w, dw)
    bc_mixed = [DirichletBC(V_mixed.sub(1), Constant(0), "on_boundary"),
                DirichletBC(V_mixed.sub(2), Constant(0), "on_boundary")]
    
    errors = []
    for i in range(max_iter):
        A, b = assemble_system(J, -F, bc_mixed)
        delta_w = Function(V_mixed)
        solve(A, delta_w.vector(), b)
        w.vector().axpy(1.0, delta_w.vector())
        
        # Считаем относительную ошибку
        rel_err = delta_w.vector().norm("l2") / (w.vector().norm("l2") + 1e-12)
        errors.append(rel_err)
        
        if rel_err < tol: break
    return errors

# --- МЕТОД РАСЩЕПЛЕНИЯ (ВАБИЩЕВИЧ) ---
def solve_vabishchevich(rho_n, u_n, tau, tol):
    rho_k = Function(V_rho)
    rho_k.assign(rho_n)
    u_k = Function(V_u)
    u_k.assign(u_n)
    
    errors = []
    bc_u = DirichletBC(V_u, Constant((0, 0)), "on_boundary")
    
    # Определяем пробные функции ОДИН РАЗ вне цикла
    r_trial = TrialFunction(V_rho)
    u_trial = TrialFunction(V_u)
    phi = TestFunction(V_rho)
    psi = TestFunction(V_u)
    
    for k in range(max_iter):
        rho_old_vec = rho_k.vector().copy()
        
        # 1. Шаг по Плотности
        # Уравнение: (r - rho_n)/tau * phi * dx - r * dot(u_k, grad(phi)) * dx = 0
        # Здесь r_trial - это искомое (v_1), phi - тест (v_0)
        a_r = (r_trial / tau) * phi * dx - r_trial * dot(u_k, grad(phi)) * dx
        L_r = (rho_n / tau) * phi * dx
        
        solve(a_r == L_r, rho_k)
        
        # 2. Шаг по Скорости
        p_k = c**2 * rho_k + (gamma - 1) * rho_k**gamma
        # Линеаризуем конвекцию, используя u_k из предыдущей итерации
        F_u = (rho_k * dot(u_trial, psi) - rho_n * dot(u_n, psi)) / tau * dx \
              - inner(rho_k * outer(u_trial, u_k), grad(psi)) * dx \
              - p_k * div(psi) * dx
        
        solve(lhs(F_u) == rhs(F_u), u_k, bc_u)
        
        # Считаем относительное приращение
        diff = rho_k.vector() - rho_old_vec
        rel_err = diff.norm("l2") / (rho_k.vector().norm("l2") + 1e-12)
        errors.append(rel_err)
        
        if rel_err < tol and k > 0:
            break
            
    return errors
# Запуск и сравнение
err_newton = solve_newton(rho_n, u_n, tau, tol)
err_vabish = solve_vabishchevich(rho_n, u_n, tau, tol)

# Вывод таблицы
print(f"{'Iter':<5} | {'Newton RelErr':<15} | {'Vabish RelErr':<15}")
for i in range(max(len(err_newton), len(err_vabish))):
    n_val = f"{err_newton[i]:.2e}" if i < len(err_newton) else "---"
    v_val = f"{err_vabish[i]:.2e}" if i < len(err_vabish) else "---"
    print(f"{i:<5} | {n_val:<15} | {v_val:<15}")

# График
plt.semilogy(err_newton, 'bo-', label='Newton (Full Implicit)')
plt.semilogy(err_vabish, 'rs-', label='Vabishchevich (Splitting)')
plt.axhline(y=tol, color='g', linestyle='--', label='Tolerance')
plt.xlabel('Iteration'); plt.ylabel('Relative Error'); plt.legend(); plt.grid(); plt.show()