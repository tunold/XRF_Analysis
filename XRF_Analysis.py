import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import xraylib
import xml.etree.ElementTree as ET
from math import radians, cos, exp
from itertools import cycle
from collections import defaultdict


# ---------- Utility functions ----------

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


def parse_spx_spectrum_with_metadata(file_path):
    tree = ET.parse(file_path)
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


def format_composition_as_formula(composition):
    total = sum(composition.values())
    scaled = {el: val * 5 / total for el, val in composition.items()}
    formula = ''.join(f"{el}{scaled[el]:.2f}" for el in ['Cs', 'Sn', 'Pb', 'I', 'Br'] if el in scaled)
    return formula, scaled


def summarize_composition_sites(scaled):
    A = scaled.get('Cs', 0)
    B = scaled.get('Sn', 0) + scaled.get('Pb', 0)
    X = scaled.get('I', 0) + scaled.get('Br', 0)
    st.text("Site analysis (ABX3):")
    st.text(f"  A-site (Cs): {A:.2f}")
    st.text(f"  B-site (Sn+Pb): {B:.2f}")
    st.text(f"  X-site (I+Br): {X:.2f}")
    A = scaled.get('Cs', 0)
    B = scaled.get('Sn', 0) + scaled.get('Pb', 0)
    X = scaled.get('I', 0) + scaled.get('Br', 0)
    st.text("Site analysis(ABX3): ")
    st.text(f"  A-site (Cs): {A:.2f}")
    st.text(f"  B-site (Sn+Pb): {B:.2f}")
    st.text(f"  X-site (I+Br): {X:.2f}")
    A = scaled.get('Cs', 0)
    B = scaled.get('Sn', 0) + scaled.get('Pb', 0)
    X = scaled.get('I', 0) + scaled.get('Br', 0)
    st.text(" Site analysis(ABX3): ")
    st.text(f"  A-site (Cs): {A:.2f}")
    st.text(f"  B-site (Sn+Pb): {B:.2f}")
    st.text(f"  X-site (I+Br): {X:.2f}")

    # ---------- Streamlit App ----------
    st.title("XRF Quantification with Self-Absorption Correction")

    primary_file = st.file_uploader("Upload primary excitation spectrum (.spx)", type="spx")
    sample_file = st.file_uploader("Upload sample XRF measurement (.spx)", type="spx")

    if primary_file and sample_file:
        energy_primary, primary_counts, meta_primary = parse_spx_spectrum_with_metadata(primary_file)
    energies, counts, meta_sample = parse_spx_spectrum_with_metadata(sample_file)
    st.write("Instrument:", meta_sample['instrument'])
    st.write("Measurement date:", meta_sample['date'], meta_sample['time'])

    fig, ax = plt.subplots()
    ax.plot(energies, counts, label="Sample spectrum")
    ax.plot(energy_primary, primary_counts, label="Primary spectrum")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    ax.set_title("XRF Spectra")
    ax.legend()
    st.pyplot(fig)

    elements = st.multiselect("Select elements", ["Cs", "Pb", "Sn", "I", "Br"], default=["Cs", "Pb", "Sn", "I", "Br"])

    line_selection = {
        "Cs": ["Lβ1"],
        "Sn": ["Lα1", "Lβ1"],
        "Pb": ["Lα1", "Lβ1"],
        "I": ["Lα1"],
        "Br": ["Kα1", "Kβ1"]
    }

    initial_composition = {"Cs": 0.18, "Sn": 0.05, "Pb": 0.12, "I": 0.25, "Br": 0.40}
    density = st.number_input("Sample density [g/cm3]", value=5.0)
    thickness_nm = st.number_input("Film thickness [nm]", value=500.0)
    angle_deg = st.number_input("Detector angle [deg]", value=45.0)

    if st.button("Quantify"):
        tau_profile = []  # collect (element, line, E_line, tau_value)
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
        valid_mask = energy_primary > 0.01
        valid_energies = energy_primary[valid_mask]
        valid_primary = primary_counts[valid_mask]
        tau = []
        for E in valid_energies:
            try:
                tau.append(xraylib.CS_Photo(Z, float(E)))
            except Exception:
                tau.append(0.0)
        tau = np.array(tau)

        exc_int = np.trapz(valid_primary * tau, valid_energies)
        idx = (energies > E_line - 0.1) & (energies < E_line + 0.1)
        I = np.sum(counts[idx])
        mu_mix = compute_mu_mix(initial_composition, E_line)
        C_reabs = compute_self_absorption_correction(mu_mix, density, thickness_nm * 1e-7, angle_deg)
        tau_value = mu_mix * density * (thickness_nm * 1e-7) / cos(radians(angle_deg))
        tau_profile.append((el, line_label, E_line, tau_value))
        corrected = (I / (omega * p * exc_int)) / C_reabs
        result[el] += corrected


total = sum(result.values())
result_norm = {el: 100 * val / total for el, val in result.items()}

st.subheader("Atomic Percent (with self-absorption):")
for el, val in result_norm.items():
    st.write(f"{el}: {val:.2f} at.%")

# Display pseudo-formula and site analysis
formula_string, scaled = format_composition_as_formula(result_norm)
st.markdown(f"**Pseudo-formula:** `{formula_string}`")
summarize_composition_sites(scaled)
