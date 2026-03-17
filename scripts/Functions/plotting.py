# Functions/plotting.py
import numpy as np
from matplotlib import pyplot as plt
from Functions.analysis import specific_energy, ang_momentum_mag, swarm_dispersion
from Functions.formation_frames import lvlh_dcm_from_rv, eci_to_lvlh


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
    plt.legend(ncol=2); plt.tight_layout(); plt.show(block=False)

def plot_osculating_oe(OE_osc, sat_names, t_hours, save_dir=None):
    OE_labels = ["$a$ (m)", "$u$ (rad)", "$e_x$", "$e_y$", "$i$ (rad)", r"$\Omega$ (rad)"]
    OE_names  = ["Semi-major axis", "Arg. of latitude", "ex", "ey", "Inclination", "RAAN"]
    colors    = plt.cm.tab10(np.linspace(0, 1, len(sat_names)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Osculating Orbital Elements — All Satellites", fontsize=13)
    for j, ax in enumerate(axes.flatten()):
        for name, color in zip(sat_names, colors):
            ax.plot(t_hours, OE_osc[name][j, :], label=name, color=color)
        ax.set_xlabel("$t$ (h)")
        ax.set_ylabel(OE_labels[j])
        ax.set_title(OE_names[j])
        ax.legend(fontsize=7)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/osculating_oe.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_osc_vs_mean_oe(OE_osc, OE_mean, sat_names, t_hours):
    OE_labels = ["$a$ (m)", "$u$ (rad)", "$e_x$", "$e_y$", "$i$ (rad)", r"$\Omega$ (rad)"]
    OE_names  = ["Semi-major axis", "Arg. of latitude", "ex", "ey", "Inclination", "RAAN"]

    for name in sat_names:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        fig.suptitle(f"{name} — Osculating vs Mean OE", fontsize=13)
        for j, ax in enumerate(axes.flatten()):
            ax.plot(t_hours, OE_osc[name][j, :],  label="Osculating", alpha=0.6, linewidth=0.8)
            ax.plot(t_hours, OE_mean[name][j, :], label="EU Mean",    linewidth=1.8)
            ax.set_xlabel("$t$ (h)")
            ax.set_ylabel(OE_labels[j])
            ax.set_title(OE_names[j])
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show(block=False)


def plot_delta_oe(OE_mean, sat_names, t_hours, mothership_name):
    OE_labels  = ["$a$ (m)", "$u$ (rad)", "$e_x$", "$e_y$", "$i$ (rad)", r"$\Omega$ (rad)"]
    OE_names   = ["Semi-major axis", "Arg. of latitude", "ex", "ey", "Inclination", "RAAN"]
    deputies   = [n for n in sat_names if n != mothership_name]
    colors     = plt.cm.tab10(np.linspace(0, 1, len(deputies)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"ΔOE Deputies − {mothership_name} (EU Mean)", fontsize=13)
    for j, ax in enumerate(axes.flatten()):
        for name, color in zip(deputies, colors):
            delta = OE_mean[name][j, :] - OE_mean[mothership_name][j, :]
            ax.plot(t_hours, delta, label=f"{name}−{mothership_name}", color=color)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("$t$ (h)")
        ax.set_ylabel("Δ " + OE_labels[j])
        ax.set_title("Δ " + OE_names[j])
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()



def plot_relative_planar(rv, ref_name, deputy_name, t_hours):

    r_ref, _ = rv[ref_name]
    r_dep, _ = rv[deputy_name]

    # Relative position in ECI (or whatever frame rv is in)
    dr = r_dep - r_ref          # shape (N, 3)
    dx = dr[:, 0] / 1e3        # X component in km
    dy = dr[:, 1] / 1e3        # Y component in km

    plt.figure(figsize=(6, 6), dpi=125)
    plt.plot(dx, dy, label=f"{deputy_name} rel. to {ref_name}")
    plt.scatter(0, 0, color='red', zorder=5, label=ref_name + " (origin)")

    # Mark start and end
    plt.scatter(dx[0],  dy[0],  marker='^', color='green', zorder=5, label='Start')
    plt.scatter(dx[-1], dy[-1], marker='s', color='black', zorder=5, label='End')

    plt.xlabel("Δx [km]")
    plt.ylabel("Δy [km]")
    plt.title(f"Relative planar trajectory: {deputy_name} w.r.t. {ref_name}")
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_relative_lvlh(rv, ref_name, deputy_name, t_hours, plane="RT"):
    """
    plane: "RT" (radial vs along-track)
           "RN" (radial vs normal)
           "TN" (along-track vs normal)
           "all" (all three as subplots)
    """
    r_ref, v_ref = rv[ref_name]
    r_dep, _     = rv[deputy_name]

    N = len(t_hours)
    dr_lvlh = np.zeros((N, 3))
    for i in range(N):
        _, C_ECI_to_LVLH = lvlh_dcm_from_rv(r_ref[i], v_ref[i])
        dr_lvlh[i] = eci_to_lvlh(C_ECI_to_LVLH, r_dep[i] - r_ref[i])

    dR = dr_lvlh[:, 0] / 1e3
    dT = dr_lvlh[:, 1] / 1e3
    dN = dr_lvlh[:, 2] / 1e3

    axes_map = {
        "RT": (dT, dR, "Along-track T [km]", "Radial R [km]"),
        "RN": (dN, dR, "Normal N [km]",      "Radial R [km]"),
        "TN": (dN, dT, "Normal N [km]",      "Along-track T [km]"),
    }

    def _single(ax, x, y, xlabel, ylabel):
        ax.plot(x, y, label=f"{deputy_name} rel. to {ref_name}")
        ax.scatter(0, 0, color='red',  zorder=5, label=ref_name + " (origin)")
        ax.scatter(x[0],  y[0],  marker='^', color='green', zorder=5, label='Start')
        ax.scatter(x[-1], y[-1], marker='s', color='black', zorder=5, label='End')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    if plane == "all":
        fig, axs = plt.subplots(1, 3, figsize=(16, 5), dpi=125)
        fig.suptitle(f"Relative LVLH trajectory: {deputy_name} w.r.t. {ref_name}")
        for ax, (key, (x, y, xl, yl)) in zip(axs, axes_map.items()):
            _single(ax, x, y, xl, yl)
            ax.set_title(key)
        plt.tight_layout()
    else:
        x, y, xl, yl = axes_map[plane]
        fig, ax = plt.subplots(figsize=(7, 7), dpi=125)
        ax.set_title(f"Relative LVLH trajectory ({plane}): {deputy_name} w.r.t. {ref_name}")
        _single(ax, x, y, xl, yl)
        plt.tight_layout()

    plt.show()


def plot_swarm_dispersion(rv, sat_names, t_hours, mothership_name):
    dispersion = swarm_dispersion(rv, sat_names, mothership_name)
    plt.figure(figsize=(10, 4), dpi=125)
    plt.plot(t_hours, dispersion / 1e6)   # m² -> km²
    plt.xlabel("Time [hr]")
    plt.ylabel(r"$\sum \|\mathbf{r}_i - \mathbf{r}_0\|^2$ [km²]")
    plt.title("Swarm dispersion relative to LFIRE-0")
    plt.tight_layout()
    plt.show()