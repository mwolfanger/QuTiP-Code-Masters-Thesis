# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "requests>=2.34.2",
# ]
# ///
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

from scipy.stats import cauchy
from qutip import *
from qutip.solver.parallel import parallel_map


# ============================================================
# Drive
# ============================================================

def drive_coeff(t, args):

    delta_start = args["delta_start"]
    delta_end = args["delta_end"]
    T = args["T"]
    gamma = args["gamma"]


    if t < 0 or t > T:
        return 0.0

    phase = (
        delta_start * t
        + 0.5
        * (delta_end - delta_start)
        * t**2
        / T
    )

    return (
        args["drive_amp"] 
        # * 1/(gamma + 1j * delta_start) # WEIL WIR KEINEN FREQUENZSWEEP BEI JEDEM DETUNING MACHEN UND OMEGA = 0,  WIRD DELTA_START VERWENDET ABER EINGENTLICH MUSS CODE NOCH ANGEPASST WERDEN
        * np.exp(-1j * phase)
    )


def drive_coeff_conj(t, args):

    return np.conjugate(
        drive_coeff(t, args)
    )


# ============================================================
# Spinfrequenzen erzeugen
# ============================================================

def make_spin_packets(
    N_p,
    N_s,
    omega_0,
    gamma,
    dist_spin_packets
):

    packet_size = N_s // N_p

    if N_s % N_p != 0:
        raise ValueError(
            "N_p and N_s must be divisible"
        )

    omega_array = np.array([])
    j = 0

    for i in range(N_p):

        # ----------------------------------------------------
        # Positive Abweichungen von omega_0
        # ----------------------------------------------------

        delta_values = np.abs(
            cauchy.rvs(
                loc=0,
                scale=gamma * 0.0,
                size=packet_size // 2
            )
        )

        delta_values = np.sort(delta_values)

        # ----------------------------------------------------
        # Symmetrisches Array
        # ----------------------------------------------------

        omega_packet = np.concatenate([
            omega_0 - delta_values[::-1],

            np.array([omega_0])
            if packet_size % 2 == 1
            else np.array([]),

            omega_0 + delta_values
        ])

        # ----------------------------------------------------
        # Verschiebung der Spin-Pakete
        # ----------------------------------------------------

        if i != 0:

            if i % 2 == 1:

                j = j - 1

                omega_packet -= (
                    dist_spin_packets / 2
                ) * j

            else:

                omega_packet += (
                    dist_spin_packets / 2
                ) * j

        omega_array = np.concatenate(
            [omega_array, omega_packet]
        )

    return omega_array

def make_packet_freq(omega, dist_spin_packets, N_groups):

    omega_array = np.zeros(N_groups)
    j = 0 

    for i in range(N_groups):

        omega_packet = omega

        if i != 0:
            # if i % 2 == 1:
                j = j - 1
                omega_packet -= (dist_spin_packets ) * -j
            # else:
            #     omega_packet += (dist_spin_packets ) * j

        omega_array[i] = omega_packet
            
    return omega_array


# ============================================================
# NEU:
# Spins nach Frequenz gruppieren EIGENTLICH QUATSCH, WENN MAN MAKE_SPIN_PACKETS UMSCHREIBEN WÜRDE UM DIREKT DIE ZENTRALEN FREQUENZE
# ============================================================

def group_spin_frequencies(
    omega_array,
    N_spins
):
    omega_array = np.asarray(
        omega_array,
        dtype=float
    )

    n_groups = len(omega_array)

    if n_groups > N_spins:
        raise ValueError(
            "n_groups cannot be larger than "
            "the number of spins"
        )
    if N_spins % n_groups != 0:
        raise ValueError(
                "Number of Spins must be divisible by number of packets"
                )

    N_spp = N_spins // n_groups

    groups = []

    for i in range(n_groups):

        groups.append({
            "omega": omega_array[i],
            "N": N_spp,
        })

    return groups


# ============================================================
# Gruppen ausgeben
# ============================================================

def print_spin_groups(groups):

    print("--------------------------------------")
    print("SPIN GROUPS")
    print("--------------------------------------")

    for i, group in enumerate(groups):

        print(
            f"Group {i:3d}: "
            f"N = {group['N']:4d}, "
            f"omega = "
            f"{group['omega']:.6e}"
        )

    print("--------------------------------------")


# ============================================================
# Simulation
# ============================================================

def run_simulation(
    detuning,
    config,
    normalize=False
):

    # --------------------------------------------------------
    # Frequenzbereich des Drives
    # --------------------------------------------------------

    delta_start = (
        detuning
        - detuning * 0.0
    )

    delta_end = (
        detuning
        + detuning * 0.0
    )

    # --------------------------------------------------------
    # Parameter aus config
    # --------------------------------------------------------

    gamma = config["gamma"]
    g_col = config["g_col"]
    omega = config["omega"]

    a = config["a"]

    group_Jz = config["group_Jz"]
    group_Jplus = config["group_Jplus"]
    group_Jminus = config["group_Jminus"]

    group_omegas = config["group_omegas"]

    H_drive_plus = config[
        "H_drive_plus"
    ]

    H_drive_minus = config[
        "H_drive_minus"
    ]

    H_spin = config["H_spin"]
    H_int = config["H_int"]

    kappa_tot = config[
        "kappa_tot"
    ]

    k_e2 = config["k_e2"]

    times = config["times"]

    drive_args = config[
        "drive_args"
    ].copy()

    drive_args.update({
        "delta_start": delta_start,
        "delta_end": delta_end
    })

    # ========================================================
    # Hamiltonian
    # ========================================================

    # Die Cavity bleibt fest bei omega.
    H0 = (
        omega * a.dag() * a
        + H_spin
        + g_col * H_int
    )

    H = [
        H0,

        [
            H_drive_plus,
            drive_coeff
        ],

        [
            H_drive_minus,
            drive_coeff_conj
        ]
    ]

    # ========================================================
    # Collapse operators
    # ========================================================

    c_ops = [
        np.sqrt(kappa_tot) * a
    ]

    # --------------------------------------------------------
    # Dephasierung der Spin-Gruppen
    #
    # ACHTUNG:
    # Das ist eine kollektive Dephasierung und nicht exakt
    # identisch zu unabhängiger Dephasierung jedes einzelnen
    # Spins.
    # --------------------------------------------------------

    for Jz in group_Jz:

        c_ops.append(
            np.sqrt(gamma) * Jz
        )

    # ========================================================
    # Anfangszustand
    # ========================================================

    psi0 = config["psi0"]

    # ========================================================
    # Master equation
    # ========================================================

    res = mesolve(
        H,
        psi0,
        times,
        c_ops,
        e_ops=[a],
        args=drive_args
    )

    # ========================================================
    # Input / Output Signal
    # ========================================================

    v_in = np.array([

        drive_coeff(
            t,
            drive_args
        )

        - drive_coeff_conj(
            t,
            drive_args
        )

        for t in times
    ])

    v_out = (
        v_in
        + np.sqrt(k_e2)
        * res.expect[0]
    )

    trace = res.expect[0]

    # ========================================================
    # Normalisierung
    # ========================================================

    if normalize:

        if config["norm_ref_peak"] == 0:

            raise ValueError(
                "Cannot normalize: "
                "norm_ref_peak is zero."
            )

        trace /= (
            config["norm_ref_peak"]
        )

    return trace


def run_bare_cavity(
    config,
    normalize=False
):

    # --------------------------------------------------------
    # Parameter
    # --------------------------------------------------------

    omega = config["omega"]

    n_max = config["n_max"]
    times = config["times"]

    kappa_tot = config["kappa_tot"]
    k_e2 = config["k_e2"]

    k_e1 = config["k_e1"]

    drive_args = config["drive_args"]
    drive_args.update({
        "delta_start": 0,
        "delta_end": 0
    })

    # --------------------------------------------------------
    # Bare cavity operator
    # --------------------------------------------------------

    a_bare = destroy(n_max)

    # --------------------------------------------------------
    # Hamiltonian
    # --------------------------------------------------------

    H0_bare = (
        omega
        * a_bare.dag()
        * a_bare
    )

    H_drive_bare = (
        1j
        * np.sqrt(k_e1)
        * (
            a_bare.dag()
            - a_bare
        )
    )

    H_bare = [
        H0_bare,
        [
            H_drive_bare,
            drive_coeff
        ]
    ]

    # --------------------------------------------------------
    # Collapse operators
    # --------------------------------------------------------

    c_ops_bare = [
        np.sqrt(kappa_tot)
        * a_bare
    ]

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    psi0_bare = basis(
        n_max,
        0
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    res = mesolve(
        H_bare,
        psi0_bare,
        times,
        c_ops_bare,
        e_ops=[a_bare],
        args=drive_args
    )

    # --------------------------------------------------------
    # Output signal
    # --------------------------------------------------------

    v_in = np.array([
        drive_coeff(
            t,
            drive_args
        )
        for t in times
    ])

    v_out = (
        v_in
        + np.sqrt(k_e2)
        * res.expect[0]
    )

    trace = np.abs(
        v_out
    )

    return trace

# ============================================================
# Main
# ============================================================

def main(n_processes):

    # ========================================================
    # Plot settings
    # ========================================================

    np.set_printoptions(
        threshold=sys.maxsize
    )

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 16,
        "figure.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 7,
        "legend.title_fontsize": 9,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        })

    # ========================================================
    # Simulation Setup
    # ========================================================

    # --------------------------------------------------------
    # Systemparameter
    # --------------------------------------------------------

    g_col = (
         0.5
        * 1e6
    )

    k_e2 = (
         0.00300024
        * 1e6
    )

    k_e1 = (
         0.00116125
        * 1e6
    )

    k_ext = (
        k_e1 + k_e2
    )

    kappa_tot = (
         0.0804874
        * 1e6
    )

    k_int = (
        kappa_tot - k_ext
    )

    gamma = (
         0.45
        * 1e6
    )

    # ========================================================
    # Drive
    # ========================================================

    drive_amp = 1
    T = 180e-6

    drive_args = {
        "T": T,
        "drive_amp": drive_amp,
        "gamma": gamma
    }

    # ========================================================
    # Simulationszeit
    # ========================================================

    t_max = 240e-6

    times = np.linspace(
        0,
        t_max,
        10000
    )

    # ========================================================
    # Frequenzen
    # ========================================================

    omega = 0
    omega_0 = 0

    # ========================================================
    # Hilbertraum
    # ========================================================

    N_max = 3
    n_max = 8

    # --------------------------------------------------------
    # NEU:
    # Anzahl der Frequenzgruppen
    # --------------------------------------------------------

    N_groups = 3

    # ========================================================
    # Spinverteilung
    # ========================================================

    dist_spin_packets = (
            0.3 * 1e6
    )

    omega_array = make_packet_freq(
            omega_0,
            dist_spin_packets,
            N_groups
    )

    print("--------------------------------------")
    print("ORIGINAL SPIN DISTRIBUTION")
    print("--------------------------------------")
    print(omega_array)

    # ========================================================
    # NEU:
    # Frequenzen gruppieren
    # ========================================================

    spin_groups = group_spin_frequencies(
        omega_array,
        N_max
    )

    print_spin_groups(
        spin_groups
    )

    # ========================================================
    # QuTiP Operatoren
    # ========================================================

    # --------------------------------------------------------
    # Dimensionen der Gruppen
    # --------------------------------------------------------

    group_dims = [
        group["N"] + 1
        for group in spin_groups
    ]

    print("--------------------------------------")
    print("GROUP DIMENSIONS")
    print("--------------------------------------")
    print(group_dims)

    # ========================================================
    # Cavity Operator
    # ========================================================

    a = tensor(
        destroy(n_max),
        *[
            qeye(dim)
            for dim in group_dims
        ]
    )

    # ========================================================
    # Kollektive Spinoperatoren
    # ========================================================

    group_Jplus = []
    group_Jminus = []
    group_Jz = []

    for i, group in enumerate(
        spin_groups
    ):

        N_group = group["N"]

        J = N_group / 2

        # ----------------------------------------------------
        # Operatoren im lokalen Gruppenraum
        # ----------------------------------------------------

        Jplus_local = jmat(
            J,
            "+"
        )

        Jminus_local = jmat(
            J,
            "-"
        )

        Jz_local = jmat(
            J,
            "z"
        )

        # ----------------------------------------------------
        # Operatoren auf gesamten Hilbertraum
        # ----------------------------------------------------

        operators_plus = [
            qeye(dim)
            for dim in group_dims
        ]

        operators_minus = [
            qeye(dim)
            for dim in group_dims
        ]

        operators_z = [
            qeye(dim)
            for dim in group_dims
        ]

        operators_plus[i] = (
            Jplus_local
        )

        operators_minus[i] = (
            Jminus_local
        )

        operators_z[i] = (
            Jz_local
        )

        group_Jplus.append(
            tensor(
                qeye(n_max),
                *operators_plus
            )
        )

        group_Jminus.append(
            tensor(
                qeye(n_max),
                *operators_minus
            )
        )

        group_Jz.append(
            tensor(
                qeye(n_max),
                *operators_z
            )
        )

    # ========================================================
    # Spin-Hamiltonian
    # ========================================================

    H_spin = 0

    for i, group in enumerate(
        spin_groups
    ):

        # ----------------------------------------------------
        # WICHTIG:
        #
        # Der alte Code verwendete
        #
        #     omega * sigma_z
        #
        # mit sigma_z = diag(+1,-1).
        #
        # jmat(...,"z") besitzt dagegen Eigenwerte
        #
        #     -J,...,+J.
        #
        # Daher Faktor 2.
        # ----------------------------------------------------

        H_spin += (
            2
            * group["omega"]
            * group_Jz[i]
        )

    # ========================================================
    # Cavity-Spin Interaktion
    # ========================================================

    H_int = 0

    for i in range(
        N_groups
    ):

        H_int += (
            a.dag()
            * group_Jminus[i]
            +
            a
            * group_Jplus[i]
        )

    # ========================================================
    # Drive Operatoren
    # ========================================================

    H_drive_plus = (
        1j
        * np.sqrt(k_e1)
        * a.dag()
    )

    H_drive_minus = (
        -1j
        * np.sqrt(k_e1)
        * a
    )

    # ========================================================
    # Anfangszustand
    # ========================================================

    # Alle Spins im unteren Zustand m=-J
    group_ground_states = []

    for group in spin_groups:

        J = group["N"] 

        group_ground_states.append(
            basis(
                int( J + 1),
                0
            )
        )

    psi0 = tensor(
        basis(n_max, 0),
        *group_ground_states
    )

    # ========================================================
    # Gemeinsame Simulationsparameter
    # ========================================================

    config = {

        "gamma": gamma,
        "g_col": g_col,
        "omega": omega,

        "a": a,

        "group_Jz": group_Jz,
        "group_Jplus": group_Jplus,
        "group_Jminus": group_Jminus,

        "group_omegas": [
            group["omega"]
            for group in spin_groups
        ],

        "H_drive_plus":
            H_drive_plus,

        "H_drive_minus":
            H_drive_minus,

        "H_spin":
            H_spin,

        "H_int":
            H_int,

        "kappa_tot":
            kappa_tot,

        "k_e2":
            k_e2,

        "k_e1":
            k_e1,

        "times":
            times,

        "drive_args":
            drive_args,

        "psi0":
            psi0,

        "N_max":
            N_max,

        "N_groups":
            N_groups,

        "n_max":
            n_max,
    }

    # ========================================================
    # Referenz: nackte Kavität
    # ========================================================

    bare_cavity = run_bare_cavity(
        config=config,
        normalize=False
    )

    norm_ref_peak = np.max(
        np.abs(
            bare_cavity
        )
    )

    config[
        "norm_ref_peak"
    ] = norm_ref_peak

    # ========================================================
    # Detuning-Bereich
    # ========================================================

    detuning_values = (
         np.linspace(
            -3e6,
            0,#3e6,
            201
        )
    )

    # ========================================================
    # Parallelisierung
    # ========================================================

    print()
    print("--------------------------------------")
    print(
        f"Starting "
        f"{len(detuning_values)} simulations"
    )

    print(
        f"Parallel processes: "
        f"{n_processes}"
    )

    results = parallel_map(
        run_simulation,
        detuning_values,
        task_args=(config,),
        task_kwargs={
            "normalize": True
        },
        map_kw={
            "num_cpus":
                n_processes
        },
        progress_bar="text",
    )

    Z = np.asarray(
        results
    )

    print()
    print(
        "Z shape:",
        Z.shape
    )

    # ========================================================
    # Contourplot
    # ========================================================

    t_start = T 

    start_idx = np.searchsorted(
        times,
        t_start
    )

    times_after = times[
        start_idx:
    ]

    Z_after = Z[
        :,
        start_idx:
    ]

    T_after, D = np.meshgrid(
        times_after,
        detuning_values
    )

    plt.figure(
        1,
        figsize=(6, 4)
    )

    cont = plt.contourf(
        T_after * 1e6,
        D ,
        np.maximum(
            np.abs(Z_after),
            1e-12
        ),
        levels=100,
        cmap="viridis"
    )
    fig = plt.gcf()
    fig.set_size_inches(17 / 2.54, 17 / 2.54)

    plt.xlabel(
        "Time (us)",
        fontsize=18, 
        fontweight='bold',
        fontname='Arial'
    )

    plt.ylabel(
        r"Detuning (MHz)",
        fontsize=18, 
        fontweight='bold',
        fontname='Arial'
    )

    # plt.title(
    #     "Response after drive pulse"
    # )

    cbar = plt.colorbar(
        cont
    )
    cbar.set_label(
        label="Normalized amplitude",
        fontweight='bold')

    plt.tight_layout()

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    cbar.ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    plt.savefig(
        "whole_spectrum.png",
        dpi=300,
        #bbox_inches="tight"
    )

    # ========================================================
    # Detuning-Spektrum
    # ========================================================

    t_cut = T 

    time_idx = np.argmin(
        np.abs(
            times - t_cut
        )
    )

    spectrum = Z[
        :,
        time_idx
    ]

    plt.figure(
        2,
        figsize=(6, 4)
    )

    plt.plot(
        detuning_values,
        np.log(np.abs(spectrum)),
        linewidth=1.5
    )
    fig = plt.gcf()
    ax = plt.gca()


    plt.xlabel(
        r"Detuning (MHz)",
        fontsize=18, 
        fontweight='bold',
        fontname='Arial'
    )

    plt.ylabel(
        r"Normalized amplitude",
        fontsize=18, 
        fontweight='bold',
        fontname='Arial'
    )

    # plt.title(
    #     f"Detuning spectrum at "
    #     f"t = "
    #     f"{times[time_idx] * 1e6:.1f}"
    #     f" $\\mu s$"
    # )

    plt.grid(True)

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    plt.tight_layout()
    fig.set_size_inches(17 / 2.54, 13 / 2.54)
    fig.set_facecolor('white')
    # Achsenformatierung
    ax.minorticks_on()
    # Major Ticks
    ax.tick_params(
        axis='both', which='major',
        direction='in', length=5.76, width=2,
        top=True,right=True
    )
    
    # Minor Ticks
    ax.tick_params(
        axis='both', which='minor',
        direction='in', length=3, width=1.5,
        bottom=True, top=True, left=True, right=True
    ) 
    # Rahmen und Gitter
    ax.grid(False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Schriftart der Achsenticks
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('demibold')
    
    # Schriftart der Achsenbeschriftungen
    ax.xaxis.label.set_fontname('Arial')
    ax.yaxis.label.set_fontname('Arial')
    ax.xaxis.label.set_fontweight('demibold')
    ax.yaxis.label.set_fontweight('demibold')

    plt.savefig(
        "detuning_spectrum.png",
        dpi=300,
        #bbox_inches="tight"
    )

    # ========================================================
    # Zeit-Spektrum
    # ========================================================

    d_cut = (
        g_col
    )

    detuning_idx = np.argmin(
        np.abs(
            detuning_values
            - d_cut
        )
    )

    spectrum = Z[
        detuning_idx,
        :
    ]

    plt.figure(
        3,
        figsize=(6, 4)
    )

    plt.plot(
        times * 1e6,
        np.abs(spectrum),
        linewidth=1.5
    )
    fig = plt.gcf()
    ax = plt.gca()

    plt.xlabel(
        r"Time (us)",
        fontsize=18, 
        fontweight='bold',
        fontname='Arial'
    )

    plt.ylabel(
        r"Normalized amplitude",
        fontsize=18, 
        fontweight='bold',
        fontname='Arial'
    )

    # plt.title(
    #     f"Time dependent spectrum at "
    #     f"$\\Delta$ = "
    #     f"{d_cut / 1e6:.1f}"
    #     f" MHz"
    # )

    plt.grid(True)

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    plt.tight_layout()
    fig.set_size_inches(17 / 2.54, 17 / 2.54)
    fig.set_facecolor('white')
    # Achsenformatierung
    ax.minorticks_on()
    # Major Ticks
    ax.tick_params(
        axis='both', which='major',
        direction='in', length=5.76, width=2,
        top=True,right=True
    )
    
    # Minor Ticks
    ax.tick_params(
        axis='both', which='minor',
        direction='in', length=3, width=1.5,
        bottom=True, top=True, left=True, right=True
    ) 
    # Rahmen und Gitter
    ax.grid(False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Schriftart der Achsenticks
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('demibold')
    
    # Schriftart der Achsenbeschriftungen
    ax.xaxis.label.set_fontname('Arial')
    ax.yaxis.label.set_fontname('Arial')
    ax.xaxis.label.set_fontweight('demibold')
    ax.yaxis.label.set_fontweight('demibold')


    plt.savefig(
        "time_spectrum.png",
        dpi=300,
        #bbox_inches="tight"
    )


    # ========================================================
    # Contourplot
    # ========================================================

    t_start = 0 

    start_idx = np.searchsorted(
        times,
        t_start
    )

    times_after = times[
        start_idx:
    ]

    Z_after = Z[
        :,
        start_idx:
    ]

    T_after, D = np.meshgrid(
        times_after,
        detuning_values
    )

    plt.figure(
        4,
        figsize=(6, 4)
    )

    cont = plt.contourf(
        T_after * 1e6,
        D ,
        np.maximum(
            np.abs(Z_after),
            1e-12
        ),
        levels=100,
        cmap="nipy_spectral"
        #cmap="viridis"
    )
    fig = plt.gcf()
    fig.set_size_inches(17 / 2.54, 13 / 2.54)

    plt.xlabel(
        "Time (us)",
        fontsize=18, 
        fontweight='bold',
        fontname='Arial'
    )

    plt.ylabel(
        r"Detuning (MHz)",
        fontsize=18, 
        fontweight='bold',
        fontname='Arial'
    )

    # plt.title(
    #     "Response after drive pulse"
    # )

    cbar = plt.colorbar(
        cont
    )
    cbar.set_label(
        label="Normalized amplitude",
        fontweight='bold')

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    cbar.ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    plt.tight_layout()

    plt.savefig(
        "response_after_drive.png",
        dpi=300,
        #bbox_inches="tight"
    )

    plt.show()

# ============================================================
# Programmstart
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "QuTiP simulation with "
            "grouped spin ensemble"
        )
    )

    parser.add_argument(
        "n_processes",
        type=int,
        nargs="?",
        default=1,
        help=(
            "Number of processes "
            "(default: 1)"
        )
    )

    args = parser.parse_args()

    if args.n_processes < 1:

        parser.error(
            "Number of processes "
            "must be at least 1"
        )

    main(
        args.n_processes
    )
