import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
print(jax.local_device_count(),flush=True)
print(jax.devices(),flush=True)
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from functools import partial
import matplotlib.pyplot as plt

omega_0 = 5.0
F_m = 1.0
omega_F = 0.1

#Analytical Solution
@partial(jit)
def analytic_solution(t_coordinates:jnp.ndarray,u0=jnp.array([0.0,0.0],dtype=jnp.float64)):
    x = (F_m/(omega_0**2 - omega_F**2)) * (jnp.cos(omega_F*t_coordinates) - jnp.cos(omega_0*t_coordinates))+ u0[0]*jnp.cos(omega_0*t_coordinates) + (u0[1]/omega_0)*jnp.sin(omega_0*t_coordinates)
    v = (F_m/(omega_0**2 - omega_F**2)) * (-omega_F*jnp.sin(omega_F*t_coordinates) + omega_0*jnp.sin(omega_0*t_coordinates))+ u0[0]*(-omega_0)*jnp.sin(omega_0*t_coordinates) + u0[1]*jnp.cos(omega_0*t_coordinates)
    return x, v

@partial(jit)
def get_total_energy(x:jnp.ndarray, v:jnp.ndarray):
    return 0.5*(v**2 + (omega_0**2)*(x**2))

@partial(jit)
def get_L2_error(u_numerical:jnp.ndarray, u_analytic:jnp.ndarray):
    norm=jnp.array([omega_0,1])   #weighting the position error by omega_0 to make it comparable to the velocity error
    return jnp.linalg.norm((u_numerical - u_analytic)*norm[:,jnp.newaxis], axis=0)


#u=[x, v]
#RHS of coupled ODEs
@partial(jit)
def f_acceleration(x:jnp.ndarray, t:jnp.ndarray):
    return F_m*jnp.cos(omega_F*t) - (omega_0**2)*x

@partial(jit)
def f_u(u:jnp.ndarray, t_coordinates:jnp.ndarray):
    v = u[1]
    a = f_acceleration(u[0], t_coordinates)
    return jnp.array([v, a])

@partial(jit,static_argnames=['f_u', 'method_name'])
def solver(f_u, u0, t_coordinates,method_name='Forward_Euler'):
    dt_values = t_coordinates[1:] - t_coordinates[:-1]
    if method_name == 'Forward_Euler':
        def step_fn(ui, i):
            t = t_coordinates[i]
            dt = dt_values[i]
            ui_next = ui + dt * f_u(ui, t)
            return ui_next, ui_next  # Carry the new state, and also output it
    elif method_name == 'Symplectic_Euler':
        def step_fn(ui, i):
            x, v = ui
            t = t_coordinates[i]
            a = f_acceleration(x, t)
            v_next = v + a * dt_values[i]
            x_next = x + v_next * dt_values[i]
            ui_next = jnp.array([x_next, v_next])
            return ui_next, ui_next
    elif method_name == 'RK4':
        def step_fn(ui, i):
            t = t_coordinates[i]
            dt = dt_values[i]
            k1 = f_u(ui, t)
            k2 = f_u(ui + k1 * dt/2, t + dt/2)
            k3 = f_u(ui + k2 * dt/2, t + dt/2)
            k4 = f_u(ui + k3 * dt, t + dt)
            ui_next = ui + (k1 + 2*k2 + 2*k3 + k4) * dt/6
            return ui_next, ui_next
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    indices = jnp.arange(len(t_coordinates) - 1)
    u_final, u_history = scan(step_fn, u0, indices)
    return jnp.concatenate([u0[None, :], u_history], axis=0).T

class PS3_Problem2:
    def __init__(self,u0=jnp.array([0.0,0.0],dtype=jnp.float64)):
        self.omega_0 = omega_0
        self.F_m = F_m
        self.omega_F = omega_F
        self.u0 = u0
        print(f"Initialized PS3_Problem2 with omega_0={self.omega_0}, F_m={self.F_m}, omega_F={self.omega_F}",flush=True)
        print(f"Initial conditions: x0={self.u0[0]}, v0={self.u0[1]}",flush=True)
    def solve(self,t_stop=100,N=100000,u0=jnp.array([0.0,0.0]),method_name='Forward_Euler',plot=False):
        t_coordinates = jnp.linspace(0, t_stop, N+1,endpoint=True, dtype=jnp.float64)
        x_numerical,v_numerical = solver(f_u, u0, t_coordinates,method_name=method_name)
        x_analytic, v_analytic = analytic_solution(t_coordinates, u0)
        print(f"Solve with {method_name}, time range: [0, {t_stop}], number of steps: {N}",flush=True)
        if plot:
            self.compare_plot(jnp.array([x_numerical, v_numerical]), jnp.array([x_analytic, v_analytic]), t_coordinates, method_name)
    def compare_plot(self, u_numerical, u_analytic, t_coordinates, method_name):
        Energy_analytic=get_total_energy(u_analytic[0], u_analytic[1])
        Energy_numerical=get_total_energy(u_numerical[0], u_numerical[1])
        L2_error = get_L2_error(u_numerical, u_analytic)
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        # x(t) vs t
        ax1:plt.Axes=axes[0][0]
        ax1.plot(t_coordinates,u_numerical[0], label='Numerical')
        ax1.plot(t_coordinates,u_analytic[0], alpha=0.8, label='Analytical')
        ax1.set_xlabel('$t$',fontsize=20)
        ax1.set_ylabel('$x(t)$',fontsize=20)
        ax1.set_title("x(t) vs t",fontsize=20)
        ax1.legend()

        # v(t) vs x(t)
        ax2:plt.Axes=axes[0][1]
        ax2.plot(u_numerical[1],omega_0*u_numerical[0], alpha=0.8, label='Numerical')
        ax2.plot(u_analytic[1], omega_0*u_analytic[0], alpha=0.8, label='Analytical')
        ax2.scatter(x=omega_0*self.u0[0], y=self.u0[1], color='red', label='Initial Point',linewidths=5, s=100, zorder=5)
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.set_xlabel('$ω×x(t)$',fontsize=20)
        ax2.set_ylabel('$v(t)$',fontsize=20)
        ax2.set_title("v(t) vs x(t)",fontsize=20)
        ax2.legend()

        # L2 Error 
        ax3:plt.Axes=axes[1][0]
        ax3.plot(t_coordinates, L2_error, label='$L_2$ Error')
        ax3.set_xlabel('t',fontsize=20)
        ax3.set_ylabel('Error',fontsize=20)
        ax3.set_yscale('log')
        ax3.set_title("$L_2$ Error",fontsize=20)

        # Energy vs time
        ax4:plt.Axes=axes[1][1]
        ax4.plot(t_coordinates, Energy_numerical, label='Numerical')
        ax4.plot(t_coordinates, Energy_analytic, alpha=0.8, label='Analytical')
        ax4.set_xlabel('t',fontsize=20)
        ax4.set_ylabel('Energy',fontsize=20)
        ax4.set_title("Energy vs Time",fontsize=20)
        ax4.legend()

        fig.suptitle(f"Comparison of Numerical and Analytical Solutions using {method_name}", fontsize=24)
        plt.tight_layout()
        plt.show()




