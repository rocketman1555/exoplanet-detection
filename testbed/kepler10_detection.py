"""
Exoplanet Transit Finder - Project Skeleton
Author: Patrick Collins
"""

# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve, LightCurveCollection
from astropy.timeseries import BoxLeastSquares

# === Step 1: Load Data ===
def load_lightcurve(target="Kepler-10", mission="Kepler"):
    """
    Download and return light curve data for a target star.
    target: string, e.g. "Kepler-10"
    mission: "Kepler" or "TESS"
    """
    # Extract the light curve files
    lc_search_results = search_lightcurve(target, mission=mission, exptime=1800)

    # Remove NaNs, normalize, and append the light curve files together
    lc_combined = []
    for i in range(0, min(20, len(lc_search_results))):
        lc = lc_search_results[i].download()
        lc = lc.remove_nans()
        lc = lc.normalize()
        lc_combined.append(lc)
    
    # Arrange the light curve files into a collection, stitch them together
    lcc = LightCurveCollection(lc_combined)
    lc = lcc.stitch()
    
    print(lc)
    
    return lc.time.value, lc.flux.value

# === Step 2: Preprocess ===
def preprocess_lightcurve(time, flux):
    """
    Normalize and detrend light curve.
    """
    flux = flux / np.median(flux)            # normalize
    # TODO: implement detrending (e.g., Savitzky-Golay filter)
    return time, flux

# === Step 3: Transit Detection (Box Least Squares) ===
def detect_transits(time, flux):
    """
    Run a Box Least Squares periodogram to detect transit candidates.
    """
    # Define trial periods (days)
    periods = np.linspace(0.5, 20, 30000)
    durations = np.linspace(0.05, 0.3, 25)   # in days

    bls = BoxLeastSquares(time, flux)
    results = bls.power(periods, durations)

    # Find strongest candidate
    best_idx = np.argmax(results.power)
    best_period = results.period[best_idx]
    best_t0 = results.transit_time[best_idx]
    best_duration = results.duration[best_idx]

    return best_period, best_t0, best_duration, results

# === Step 4: Validation ===
def validate_transits(time, flux, period, t0, duration):
    """
    Phase-fold the light curve and visualize the transit.
    """
    folded_time = ((time - t0 + 0.5*period) % period) - 0.5*period
    plt.figure(figsize=(10,5))
    plt.scatter(folded_time, flux, s=1, color='black')
    plt.xlabel("Phase (days)")
    plt.ylabel("Normalized Flux")
    plt.title(f"Candidate Transit: Period = {period:.4f} days")
    plt.show()

# === Step 5: Main Script ===
if __name__ == "__main__":
    # Load
    time, flux = load_lightcurve("Kepler-10")

    # Preprocess
    #time, flux = preprocess_lightcurve(time, flux)

    # Detect
    period, t0, duration, results = detect_transits(time, flux)

    # Validate
    validate_transits(time, flux, period, t0, duration)
