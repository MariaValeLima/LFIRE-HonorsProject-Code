# Functions/plotting.py
import numpy as np
from matplotlib import pyplot as plt
from Functions.analysis import specific_energy, ang_momentum_mag

def plot_eci_3d(rv, sat_names, title="ECI trajectories"):
    fig = plt.figure(figsize=(6, 6), dpi=125)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    for name in sat_names:
        r, _ = rv[name]
        ax.plot(r[:,0], r[:,1], r[:,2], label=name)
    ax.scatter(0,0,0, label="Earth", marker="o")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_box_aspect((1,1,1))
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)

def plot_radius_norm(rv, sat_names, t_hours):
    plt.figure(figsize=(10,4), dpi=125)
    for name in sat_names:
        r, _ = rv[name]
        plt.plot(t_hours, np.linalg.norm(r, axis=1)/1e3, label=name)
    plt.xlabel("Time [hr]"); plt.ylabel("||r|| [km]")
    plt.title("Radius magnitude vs time")
    plt.legend(ncol=2); plt.tight_layout(); plt.show(block=False)

def plot_speed_norm(rv, sat_names, t_hours):
    plt.figure(figsize=(10,4), dpi=125)
    for name in sat_names:
        _, v = rv[name]
        plt.plot(t_hours, np.linalg.norm(v, axis=1), label=name)
    plt.xlabel("Time [hr]"); plt.ylabel("||v|| [m/s]")
    plt.title("Speed magnitude vs time")
    plt.legend(ncol=2); plt.tight_layout(); plt.show(block=False)

def plot_components(rv, sat_names, t_hours):
    fig, axs = plt.subplots(2, 3, figsize=(12,7), sharex=True, dpi=125)
    for name in sat_names:
        r, v = rv[name]
        axs[0,0].plot(t_hours, r[:,0]/1e3, label=name)
        axs[0,1].plot(t_hours, r[:,1]/1e3)
        axs[0,2].plot(t_hours, r[:,2]/1e3)
        axs[1,0].plot(t_hours, v[:,0], label=name)
        axs[1,1].plot(t_hours, v[:,1])
        axs[1,2].plot(t_hours, v[:,2])
    axs[0,0].set_ylabel("x [km]"); axs[0,1].set_ylabel("y [km]"); axs[0,2].set_ylabel("z [km]")
    axs[1,0].set_ylabel("vx [m/s]"); axs[1,1].set_ylabel("vy [m/s]"); axs[1,2].set_ylabel("vz [m/s]")
    for j in range(3): axs[1,j].set_xlabel("Time [hr]")
    axs[0,0].legend(ncol=2)
    plt.tight_layout(); plt.show(block=False)

def plot_energy(rv, sat_names, t_hours, mu):
    plt.figure(figsize=(10,4), dpi=125)
    for name in sat_names:
        r, v = rv[name]
        plt.plot(t_hours, specific_energy(r, v, mu), label=name)
    plt.xlabel("Time [hr]"); plt.ylabel("Specific energy ε [J/kg]")
    plt.title("Specific mechanical energy")
    plt.legend(ncol=2); plt.tight_layout(); plt.show(block=False)

def plot_ang_momentum(rv, sat_names, t_hours):
    plt.figure(figsize=(10,4), dpi=125)
    for name in sat_names:
        r, v = rv[name]
        plt.plot(t_hours, ang_momentum_mag(r, v), label=name)
    plt.xlabel("Time [hr]"); plt.ylabel("||h|| [m²/s]")
    plt.title("Specific angular momentum magnitude")
    plt.legend(ncol=2); plt.tight_layout(); plt.show(block=False)

def plot_separation_to_mothership(rv, sat_names, t_hours, mothership_name):
    rC, _ = rv[mothership_name]
    plt.figure(figsize=(10,4), dpi=125)
    max_sep = 0.0
    for name in sat_names:
        if name == mothership_name:
            continue
        r, _ = rv[name]
        sep_km = np.linalg.norm(r - rC, axis=1)/1e3
        max_sep = max(max_sep, float(np.max(sep_km)))
        plt.plot(t_hours, sep_km, label=name)
    plt.xlabel("Time [hr]"); plt.ylabel("Separation [km]")
    plt.title(f"Separation to mothership (max ~ {max_sep:.2f} km)")
    plt.legend(ncol=2); plt.tight_layout(); plt.show()
