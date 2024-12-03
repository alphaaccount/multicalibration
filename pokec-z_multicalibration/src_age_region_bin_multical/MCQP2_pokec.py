import cvxpy as cp
import numpy as np

# Parameters (You need to provide actual values for these)
n = 10  # Number of elements in X (xi)
m = 5   # Number of elements in support(p) (vj)
y = np.random.rand(n)  # Vector of target values y_i
v = np.random.rand(m)  # Values in support(p) (v_j)
p = np.random.rand(n)  # Probabilities p(x_i)
p_tilde = np.random.rand(n)  # Approximate probabilities tilde(p)(x_i)
alpha = 0.1  # Given constant alpha

# Decision variable z_ij
z = cp.Variable((n, m))

# Objective function: sum of (y_i - sum(z_ij * v_j))^2
objective = cp.Minimize(cp.sum(cp.square(y[:, None] - cp.sum(cp.multiply(z, v), axis=1))))

# Constraints
constraints = [
    cp.sum(z, axis=1) == 1,  # Sum over z_ij for each xi must be 1
    0 <= z, z <= 1  # z_ij values must be between 0 and 1
]

# Add the quadratic inequality constraint:
# sum over (p(x_i) - v_j)^2 * z_ij <= alpha * sum z_ij for each S
for i in range(n):
    constraint_lhs = cp.sum(cp.square(p[i] - v) * z[i, :])
    constraint_rhs = alpha * cp.sum(z[i, :])
    constraints.append(constraint_lhs <= constraint_rhs)

# Problem setup
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Output the optimal values of z
print("Optimal z:", z.value)
print("Optimal objective value:", problem.value)
