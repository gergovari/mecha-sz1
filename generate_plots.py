import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')

J = 2
K = 400
B = 25
phi_0 = 45 * np.pi / 180
Omega_0 = 2

A = np.array([[0, 1], [-K/J, -B/J]])
x0 = np.array([phi_0, Omega_0])

def get_data(dt, steps):
    t_vals = [0]
    x_fe = [x0.copy()]
    x_be = [x0.copy()]
    Phi_fe = np.eye(2) + dt * A
    Phi_be = np.linalg.inv(np.eye(2) - dt * A)
    for i in range(steps):
        t_vals.append((i+1)*dt)
        x_fe.append(Phi_fe @ x_fe[-1])
        x_be.append(Phi_be @ x_be[-1])
    return np.array(t_vals), np.array(x_fe)[:,0], np.array(x_be)[:,0]

def draw_spines(ax, xlabel=r'$t \, [\mathrm{s}]$', ylabel=r'$\varphi \, [\mathrm{rad}]$'):
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.grid(True, linestyle='-', color='#d3d3d3', zorder=0)

    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, zorder=10)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, zorder=10)
    
    ax.text(1.02, 0, xlabel, transform=ax.get_yaxis_transform(), ha='left', va='center')
    ax.text(0, 1.02, ylabel, transform=ax.get_xaxis_transform(), ha='center', va='bottom')

def create_figure(dt, steps, filename, suffix=''):
    t, x_fe, x_be = get_data(dt, steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.step(t, x_fe, where='post', color='#008080', linewidth=2, zorder=3)
    draw_spines(ax1)
    ax1.set_title(f'(a) Előretartó Euler módszer{suffix}', y=-0.15)

    ax2.step(t, x_be, where='post', color='#FF6600', linewidth=2, zorder=3)
    draw_spines(ax2)
    ax2.set_title(f'(b) Hátratatartó Euler módszer{suffix}', y=-0.15)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

create_figure(0.1, 10, 'feladat_b_10Hz.png', suffix=', $f = 10$ Hz')

create_figure(0.05, 20, 'feladat_c_20Hz.png', suffix=', $f = 20$ Hz')

create_figure(0.01, 100, 'feladat_c_100Hz.png', suffix=', $f = 100$ Hz')
