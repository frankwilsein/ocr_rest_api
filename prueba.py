import sympy as sp

# Definir las variables
x = sp.symbols('x')
p = sp.symbols('p')
EI1 = 80000
EI2 = 40000

# Definir las funciones de aproximación
phi1 = x * sp.sin(sp.pi * x / 6)
phi2 = x**2 * sp.sin(sp.pi * x / 6)

# Derivadas segundas
d2phi1_dx2 = sp.diff(phi1, x, 2)
d2phi2_dx2 = sp.diff(phi2, x, 2)

# Integrales en los dos intervalos
integral1_phi1 = sp.integrate(EI1 * d2phi1_dx2**2 - p * sp.diff(phi1, x) * phi1, (x, 0, 4))
integral2_phi1 = sp.integrate(EI2 * d2phi1_dx2**2 - p * sp.diff(phi1, x) * phi1, (x, 4, 6))

integral1_phi2 = sp.integrate(EI1 * d2phi2_dx2**2 - p * sp.diff(phi2, x) * phi2, (x, 0, 4))
integral2_phi2 = sp.integrate(EI2 * d2phi2_dx2**2 - p * sp.diff(phi2, x) * phi2, (x, 4, 6))

integral1_cross = sp.integrate(EI1 * d2phi1_dx2 * d2phi2_dx2 - p * sp.diff(phi1, x) * phi2, (x, 0, 4))
integral2_cross = sp.integrate(EI2 * d2phi1_dx2 * d2phi2_dx2 - p * sp.diff(phi1, x) * phi2, (x, 4, 6))

# Sumar ambas integrales
B_phi1_phi1 = integral1_phi1 + integral2_phi1
B_phi2_phi2 = integral1_phi2 + integral2_phi2
B_phi1_phi2 = integral1_cross + integral2_cross

print("B_phi1_phi1: ", B_phi1_phi1)
print("B_phi2_phi2: ", B_phi2_phi2)
print("B_phi1_phi2: ", B_phi1_phi2)

# Resolver para P
from sympy import Matrix, solve

# Crear la matriz
A = B_phi1_phi1
B = B_phi1_phi2
C = B_phi1_phi2  # B_phi1_phi2 es simétrico, así que C = B
D = B_phi2_phi2

# Determinante de la matriz
matrix = Matrix([[A, B], [C, D]])
det = matrix.det()

# Resolver la ecuación característica
solution = solve(det, p)
print("Carga crítica P: ", solution)
# Simplificar la expresión
import sympy as sp

# Definir la expresión compleja
expr = 1250 * sp.pi**2 * (-23652 * sp.pi**2 - 1164 * sp.pi**4 - 288 * sp.sqrt(3) * sp.pi**3 - 3645 + 3564 * sp.sqrt(3) * sp.pi + 2 * sp.sqrt(3) * sp.sqrt(-16994448 * sp.sqrt(3) * sp.pi**3 - 314928 * sp.sqrt(3) * sp.pi - 885735 + 58560 * sp.sqrt(3) * sp.pi**7 + 716904 * sp.sqrt(3) * sp.pi**5 + 51333264 * sp.pi**2 + 123200 * sp.pi**8 + 6420096 * sp.pi**6 + 76850208 * sp.pi**4)) / (2187 * (-3 + 2 * sp.pi**2))

# Simplificar la expresión
simplified_expr = sp.simplify(expr)

print("Expresión simplificada: ", simplified_expr)

# Evaluar la expresión numéricamente
numeric_result = simplified_expr.evalf()

print("Resultado numérico: ", numeric_result)

