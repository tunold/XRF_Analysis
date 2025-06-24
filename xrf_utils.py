# xrf_utils.py
import numpy as np
import xraylib
import xml.etree.ElementTree as ET
from math import radians, cos, exp

def parse_spx_spectrum_with_metadata(file):
    tree = ET.parse(file)
    root = tree.getroot()
    spectrum_header = None
    for elem in root.iter("ClassInstance"):
        if elem.attrib.get("Type") == "TRTSpectrumHeader":
            spectrum_header = elem
            break
    if spectrum_header is None:
        raise ValueError("Could not find TRTSpectrumHeader block.")
    calib_abs = float(spectrum_header.find("CalibAbs").text)
    calib_lin = float(spectrum_header.find("CalibLin").text)
    channel_count = int(spectrum_header.find("ChannelCount").text)
    date = spectrum_header.findtext("Date", default="Unknown")
    time = spectrum_header.findtext("Time", default="Unknown")
    excitation_energy = spectrum_header.findtext("EnergyHigh", default=None)
    instrument_name = root.attrib.get("Name", "Unknown")
    counts_text = root.find(".//Channels").text
    try:
        counts = np.array([int(val) for val in counts_text.split(',')])
    except:
        counts = np.array([float(val) for val in counts_text.split(',')])
    counts = counts[:channel_count]
    channels = np.arange(channel_count)
    energies = calib_abs + calib_lin * channels
    metadata = {
        "date": date,
        "time": time,
        "instrument": instrument_name,
        "excitation_energy_kV": float(excitation_energy) if excitation_energy else None,
        "calib_abs": calib_abs,
        "calib_lin": calib_lin,
        "channel_count": channel_count
    }
    return energies, counts, metadata

def get_line_code_and_shell(label):
    label_map = {
        "Kα1": (xraylib.KA1_LINE, xraylib.K_SHELL),
        "Kβ1": (xraylib.KB1_LINE, xraylib.K_SHELL),
        "Lα1": (xraylib.LA1_LINE, xraylib.L3_SHELL),
        "Lβ1": (xraylib.LB1_LINE, xraylib.L3_SHELL)
    }
    return label_map[label]

def compute_mu_mix(composition, energy_keV):
    mu_mix = 0
    for el, frac in composition.items():
        Z = xraylib.SymbolToAtomicNumber(el)
        mu = xraylib.CS_Total(Z, energy_keV)
        mu_mix += frac * mu
    return mu_mix

def compute_self_absorption_correction(mu_mix, density, thickness_cm, angle_deg=45):
    theta = radians(angle_deg)
    path = thickness_cm / cos(theta)
    tau = mu_mix * density * path
    return (1 - exp(-tau)) / tau if tau > 1e-6 else 1.0


from collections import defaultdict
import numpy as np
import xraylib
from .xrf_core import get_line_code_and_shell  # adapt if needed

def quantify_without_self_absorption(energies, counts, energy_primary, primary_counts, elements, line_selection):
    result = defaultdict(float)

    for el in elements:
        if el not in line_selection:
            continue
        Z = xraylib.SymbolToAtomicNumber(el)
        for line_label in line_selection[el]:
            line_code, shell = get_line_code_and_shell(line_label)
            E_line = xraylib.LineEnergy(Z, line_code)
            omega = xraylib.FluorYield(Z, shell)
            p = xraylib.RadRate(Z, line_code)

            # Excitation integral
            valid_mask = energy_primary > 0.01
            valid_energies = energy_primary[valid_mask]
            valid_primary = primary_counts[valid_mask]
            tau = np.array([xraylib.CS_Photo(Z, float(E)) if E > 0 else 0 for E in valid_energies])
            exc_int = np.trapz(valid_primary * tau, valid_energies)

            # Line intensity
            idx = (energies > E_line - 0.1) & (energies < E_line + 0.1)
            I = np.sum(counts[idx])

            result[el] += I / (omega * p * exc_int)

    total = sum(result.values())
    return {el: 100 * val / total for el, val in result.items()}

from math import radians, cos
from collections import defaultdict
import numpy as np
import xraylib
from .xrf_core import get_line_code_and_shell, compute_mu_mix, compute_self_absorption_correction  # adjust if needed

def quantify_with_self_absorption(energies, counts, energy_primary, primary_counts, elements, line_selection, initial_composition, density, thickness_nm, angle_deg):
    result = defaultdict(float)

    for el in elements:
        if el not in line_selection:
            continue
        Z = xraylib.SymbolToAtomicNumber(el)
        for line_label in line_selection[el]:
            line_code, shell = get_line_code_and_shell(line_label)
            E_line = xraylib.LineEnergy(Z, line_code)
            omega = xraylib.FluorYield(Z, shell)
            p = xraylib.RadRate(Z, line_code)

            # Excitation integral
            valid_mask = energy_primary > 0.01
            valid_energies = energy_primary[valid_mask]
            valid_primary = primary_counts[valid_mask]
            tau = np.array([xraylib.CS_Photo(Z, float(E)) if E > 0 else 0 for E in valid_energies])
            exc_int = np.trapz(valid_primary * tau, valid_energies)

            # Line intensity
            idx = (energies > E_line - 0.1) & (energies < E_line + 0.1)
            I = np.sum(counts[idx])

            # Self-absorption correction
            mu_mix = compute_mu_mix(initial_composition, E_line)
            thickness_cm = thickness_nm * 1e-7
            C_reabs = compute_self_absorption_correction(mu_mix, density, thickness_cm, angle_deg)

            corrected = (I / (omega * p * exc_int)) / C_reabs
            result[el] += corrected

    total = sum(result.values())
    return {el: 100 * val / total for el, val in result.items()}

