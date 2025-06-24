# xrf_utils.py
import numpy as np
import xraylib
from math import cos, radians, exp

def get_line_code_and_shell(label):
    label_map = {
        "Kα1": (xraylib.KA1_LINE, xraylib.K_SHELL),
        "Kβ1": (xraylib.KB1_LINE, xraylib.K_SHELL),
        "Lα1": (xraylib.LA1_LINE, xraylib.L3_SHELL),
        "Lβ1": (xraylib.LB1_LINE, xraylib.L3_SHELL)
    }
    return label_map[label]

def parse_spx_spectrum_with_metadata(file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(file)
    root = tree.getroot()

    spectrum_header = None
    for elem in root.iter("ClassInstance"):
        if elem.attrib.get("Type") == "TRTSpectrumHeader":
            spectrum_header = elem
            break

    calib_abs = float(spectrum_header.find("CalibAbs").text)
    calib_lin = float(spectrum_header.find("CalibLin").text)
    channel_count = int(spectrum_header.find("ChannelCount").text)

    counts_text = root.find(".//Channels").text
    counts = np.array([float(val) for val in counts_text.split(',')])[:channel_count]
    energies = calib_abs + calib_lin * np.arange(channel_count)

    metadata = {
        "date": spectrum_header.findtext("Date", default="Unknown"),
        "time": spectrum_header.findtext("Time", default="Unknown"),
        "instrument": root.attrib.get("Name", "Unknown"),
        "excitation_energy_kV": float(spectrum_header.findtext("EnergyHigh", default=0))
    }
    return energies, counts, metadata

def quantify_xrf_custom_lines_weighted(energies, counts, energy_primary, primary, elements, line_selection, delta_E=0.1):
    results = {}
    primary = np.array(primary)
    energy_primary = np.array(energy_primary)
    primary /= np.trapz(primary, energy_primary)

    for el in elements:
        if el not in line_selection:
            continue

        Z = xraylib.SymbolToAtomicNumber(el)
        weighted_vals, weights = [], []

        for line_label in line_selection[el]:
            try:
                line_code, shell = get_line_code_and_shell(line_label)
                E_line = xraylib.LineEnergy(Z, line_code)
                omega = xraylib.FluorYield(Z, shell)
                p = xraylib.RadRate(Z, line_code)
                tau = np.array([xraylib.CS_Photo(Z, E) if E > xraylib.EdgeEnergy(Z, shell) else 0 for E in energy_primary])
                exc_int = np.trapz(primary * tau, energy_primary)

                idx = (energies > E_line - delta_E) & (energies < E_line + delta_E)
                I = np.sum(counts[idx])

                corrected = I / (omega * p * exc_int)
                w = omega * p * exc_int

                weighted_vals.append(corrected)
                weights.append(w)

            except:
                continue

        if weights:
            results[el] = np.average(weighted_vals, weights=weights)

    total = sum(results.values())
    results_norm = {el: 100 * val / total for el, val in results.items()}
    return results_norm, []

def compute_mu_mix(composition, energy_keV):
    mu_mix = 0
    for el, frac in composition.items():
        Z = xraylib.SymbolToAtomicNumber(el)
        mu_mix += frac * xraylib.CS_Total(Z, energy_keV)
    return mu_mix

def compute_self_absorption_correction(mu_mix, density, thickness_cm, angle_deg=45):
    theta = radians(angle_deg)
    path = thickness_cm / cos(theta)
    tau = mu_mix * density * path
    return (1 - exp(-tau)) / tau if tau > 1e-6 else 1.0

def quantify_with_self_absorption(energies, counts, energy_primary, primary_counts, elements, line_selection, initial_composition, density, thickness_nm, angle_deg):
    results = {}
    primary = np.array(primary_counts)
    energy_primary = np.array(energy_primary)
    primary /= np.trapz(primary, energy_primary)

    for el in elements:
        if el not in line_selection:
            continue
        Z = xraylib.SymbolToAtomicNumber(el)
        for line_label in line_selection[el]:
            try:
                line_code, shell = get_line_code_and_shell(line_label)
                E_line = xraylib.LineEnergy(Z, line_code)
                omega = xraylib.FluorYield(Z, shell)
                p = xraylib.RadRate(Z, line_code)
                tau = np.array([xraylib.CS_Photo(Z, E) if E > xraylib.EdgeEnergy(Z, shell) else 0 for E in energy_primary])
                exc_int = np.trapz(primary * tau, energy_primary)

                idx = (energies > E_line - 0.1) & (energies < E_line + 0.1)
                I = np.sum(counts[idx])

                mu_mix = compute_mu_mix(initial_composition, E_line)
                C_reabs = compute_self_absorption_correction(mu_mix, density, thickness_nm * 1e-7, angle_deg)
                corrected = (I / (omega * p * exc_int)) / C_reabs
                results[el] = results.get(el, 0) + corrected
            except:
                continue

    total = sum(results.values())
    result_norm = {el: 100 * val / total for el, val in results.items()}
    return result_norm
