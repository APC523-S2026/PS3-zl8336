import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from diffrax import (
    ODETerm, SaveAt, diffeqsolve, Dopri5, Kvaerno3, 
    ConstantStepSize, PIDController,VeryChord
)
import matplotlib.pyplot as plt
def build_rate_tensor():
    # Rate constants
    k_p1 = 2e3
    k_m1 = 3e-12
    k_p2 = 2e1

    # Initialize a 5x5x5 tensor with zeros
    K = jnp.zeros((5, 5, 5))

    # Reaction 1: N2 + O -> NO + N  |  Rate = k_p1 * X_N2 * X_O (indices 2 and 1)
    K = K.at[0, 2, 1].add(k_p1)   # dX_N
    K = K.at[1, 2, 1].add(-k_p1)  # dX_O
    K = K.at[2, 2, 1].add(-k_p1)  # dX_N2
    K = K.at[4, 2, 1].add(k_p1)   # dX_NO

    # Reaction 2: NO + N -> N2 + O  |  Rate = k_m1 * X_NO * X_N (indices 4 and 0)
    K = K.at[0, 4, 0].add(-k_m1)  # dX_N
    K = K.at[1, 4, 0].add(k_m1)   # dX_O
    K = K.at[2, 4, 0].add(k_m1)   # dX_N2
    K = K.at[4, 4, 0].add(-k_m1)  # dX_NO

    # Reaction 3: N + O2 -> NO + O  |  Rate = k_p2 * X_N * X_O2 (indices 0 and 3)
    K = K.at[0, 0, 3].add(-k_p2)  # dX_N
    K = K.at[1, 0, 3].add(k_p2)   # dX_O
    K = K.at[3, 0, 3].add(-k_p2)  # dX_O2
    K = K.at[4, 0, 3].add(k_p2)   # dX_NO
    print(f'Build rate tensor K with k_p1={k_p1}, k_m1={k_m1}, k_p2={k_p2}')
    print(f'Shape of K: {K.shape}')
    return K

# Precompute the static rate tensor
K_TENSOR = build_rate_tensor()

@jax.jit
def zeldovich_mechanism(K_TENSOR, X):
    """
    Computes dX/dt
    Equation: ijk, j, k -> i
    - i: target species index (output dimension)
    - j: first reactant index
    - k: second reactant index
    """
    return jnp.einsum('ijk,j,k->i', K_TENSOR, X, X)

def zeldovich_equations(t, X, args):
    K_TENSOR = args[0]
    return zeldovich_mechanism(K_TENSOR, X)

# --- Integrator Configurations ---

# i. Explicit (RK4 equivalent) - Fixed Step
# WARNING: This will likely fail or require a step size < 1e-4 due to stiffness
dt_fixed = 1e-4 
solver_explicit = Dopri5()
step_fixed = ConstantStepSize()
# Define the internal root finder with explicit tolerances
root_finder = VeryChord(rtol=1e-6, atol=1e-9)
# ii. & iii. Implicit (DIRK/BDF2 equivalent) - Fixed Step
solver_implicit = Kvaerno3()

# iv. & v. Adaptive Time-Stepping (10% rule / PID Control)
# The PIDController dynamically scales dt based on local error estimates
step_adaptive = PIDController(rtol=1e-5, atol=1e-7, pcoeff=0.1, icoeff=0.3)

def solve_system(term,solver, stepsize_controller, dt0,t1, y0, saveat,args):
    # JIT-compile the solver for maximum performance
    @jax.jit
    def run():
        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            args=args,
            max_steps=10000000 
        )
        return sol
    return run()

class PS3_Problem4:
    def __init__(self):
        self.K_TENSOR = build_rate_tensor()
    def calculate_Jacobian(self, X0):
        jacobian_fn = jax.jit(jax.jacfwd(lambda x: zeldovich_mechanism(self.K_TENSOR, x)))
        J0 = jacobian_fn(X0)
        eigenvalues = jnp.linalg.eigvals(J0)
        stiffness= jnp.max(jnp.abs(eigenvalues)) / jnp.min(jnp.abs(eigenvalues))
        print(f"Eigenvalues: {eigenvalues}")
        print(f"Stiffness: {stiffness}")
    def solve(self, X0, t1=40.0, num_points=500, method='Dopri5',adaptive=True,plot=True):
        term = ODETerm(zeldovich_equations)
        t0=0.0
        t_save=jnp.linspace(t0, t1, num_points, endpoint=False)
        saveat = SaveAt(ts=t_save)
        solver_dict = {
            'Dopri5': Dopri5(),
            'Kvaerno3': Kvaerno3(root_finder=root_finder)
        }
        solution=solve_system(
            term=term,solver=solver_dict[method], 
            stepsize_controller=step_adaptive if adaptive else ConstantStepSize(), 
            dt0=1e-3 if adaptive else 1e-4,
            t1=t1, y0=X0, saveat=saveat,args=(self.K_TENSOR,)
            )
        if plot:
            self.plot_solution(t_save, solution,title=f'{method} {"Adaptive" if adaptive else "Fixed Step"}')
        return solution
    def plot_solution(self,t_save, solution,title=''):
        labels = ['N', 'O', 'N2', 'O2', 'NO']
        plt.figure(figsize=(10, 6))
        for i in range(5):
            plt.plot(t_save, solution.ys[:, i], label=labels[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Mole Fraction')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log') # Log scale helps visualize trace species like NO
        plt.xlim(1e-3, t_save[-1]) # Focus on the range where dynamics occur
        plt.ylim(1e-12, 1) # Adjust y-limits to capture all species
        plt.show()

