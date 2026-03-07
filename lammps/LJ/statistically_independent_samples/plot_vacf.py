import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ------------------------------
# 1. Model: damped oscillatory VACF
# ------------------------------
def damped_cos(t, A, tau, omega, phi):
    return A * np.exp(-t / tau) * np.cos(omega * t + phi)


# ------------------------------
# 2. Automatic initial guess helper
# ------------------------------
def estimate_initial_params(t, C):
    A0 = C[0]
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(C)

    if len(peaks) > 1:
        dt = t[peaks[1]] - t[peaks[0]]
        omega0 = 2 * np.pi / dt
    else:
        omega0 = 2 * np.pi / (t[-1] - t[0])

    half_idx = np.argmin(np.abs(C - C[0] / 2))
    tau0 = t[half_idx] if half_idx > 0 else (t[-1] - t[0]) / 4
    phi0 = 0
    return A0, tau0, omega0, phi0


# ------------------------------
# 3. Fitting routine
# ------------------------------
def fit_vacf(t, C):
    p0 = estimate_initial_params(t, C)
    popt, pcov = curve_fit(damped_cos, t, C, p0=p0, maxfev=20000)
    A, tau, omega, phi = popt
    return A, tau, omega, phi, popt, pcov


# ------------------------------
# 4. Main script with pastel colors
# ------------------------------
if __name__ == "__main__":
    # Base colors (darker) for data, pastel colors for fits
    base_colors = {
        "fcc": "#1f77b4",  # dark blue
        "hcp": "#d62728",  # dark red
        "bcc": "#2ca02c",  # dark green
    }

    pastel_colors = {
        "fcc": "#7eb0d5",  # pastel blue
        "hcp": "#ff9896",  # pastel red
        "bcc": "#98df8a",  # pastel green
    }

    lattices = ["fcc", "hcp", "bcc"]
    N_values = [108000, 108000, 128000]

    plt.figure(figsize=(8, 4))

    for lattice, n in zip(lattices, N_values):
        # Load VACF
        t, C = np.loadtxt(f"results/{lattice}/vacf.dat", unpack=True)
        t = t - 10000  # adjust if needed

        # Fit
        A, tau, omega, phi, popt, pcov = fit_vacf(t, C)

        print(f"--- {lattice} ---")
        print(f"A = {A:.5f}, τ = {tau:.5f}, ω = {omega:.5f}, φ = {phi:.5f}")

        C_fit = damped_cos(t, *popt)

        # Plot data (solid)
        plt.plot(
            t, C, color=base_colors[lattice], lw=2, label=f"{lattice} data, $N={n}$"
        )

        # Plot fit (dashed, pastel)
        plt.plot(
            t,
            C_fit,
            ls="--",
            color=base_colors[lattice],
            lw=2,
            alpha=0.7,
            zorder=-10,
            label=f"{lattice} fit, $\\tau_c={tau:.3f}$, $\\omega={omega:.3f}$",
        )

    plt.xlabel("$t/\\tau_{s}$")
    plt.ylabel("$C_v(t)$")
    plt.legend()
    plt.xlim(0, 60)
    plt.tight_layout()
    plt.savefig("vacf_fit.pdf")
