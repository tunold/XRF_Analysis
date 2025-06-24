# xrf_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from xrf_utils import parse_spx_spectrum_with_metadata, get_line_code_and_shell, compute_mu_mix, compute_self_absorption_correction
import xraylib

st.title("XRF Spectrum Quantification")

primary_file = st.file_uploader("Upload primary excitation spectrum (.spx)", type="spx")
sample_file = st.file_uploader("Upload sample XRF measurement (.spx)", type="spx")

if primary_file and sample_file:
    energy_primary, primary_counts, meta_primary = parse_spx_spectrum_with_metadata(primary_file)
    energies, counts, meta_sample = parse_spx_spectrum_with_metadata(sample_file)

    st.write("Instrument:", meta_sample['instrument'])
    st.write("Measurement date:", meta_sample['date'], meta_sample['time'])

    fig, ax = plt.subplots()
    ax.plot(energies, counts, label="Sample spectrum")
    #ax.plot(energy_primary, primary_counts, label="Primary spectrum")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    ax.set_title("XRF Spectra")
    ax.set_yscale('log')
    ax.legend()
    st.pyplot(fig)

    elements = st.multiselect("Select elements", ["Cs", "Pb", "Sn", "I", "Br"], default=["Cs", "Pb", "Sn", "I", "Br"])

    line_selection = {
        "Cs": ["Lβ1"],
        "Sn": ["Lα1", "Lβ1"],
        "Pb": ["Lα1", "Lβ1"],
        "I":  ["Lα1"],
        "Br": ["Kα1", "Kβ1"]
    }



    # initial_composition will be set after quantification without self-absorption
    initial_composition = None
    density = st.number_input("Sample density [g/cm³]", value=5.0, key="density_input")
    thickness_nm = st.number_input("Film thickness [nm]", value=500.0, key="thickness_input")
    angle_deg = st.number_input("Detector angle [deg]", value=45.0, key="angle_input")

    if st.button("Quantify (no self-absorption)"):
        result_noabs = defaultdict(float)
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
                        if E > 0.1:
                            tau.append(xraylib.CS_Photo(Z, float(E)))
                        else:
                            tau.append(0.0)
                    except Exception:
                        tau.append(0.0)
                tau = np.array(tau)
                exc_int = np.trapz(valid_primary * tau, valid_energies)
                idx = (energies > E_line - 0.1) & (energies < E_line + 0.1)
                I = np.sum(counts[idx])
                corrected = I / (omega * p * exc_int)
                result_noabs[el] += corrected
        total = sum(result_noabs.values())
        st.session_state['result_noabs_norm'] = {el: 100 * val / total for el, val in result_noabs.items()}

        st.subheader("Atomic Percent (no self-absorption):")
        for el, val in st.session_state['result_noabs_norm'].items():
            st.write(f"{el}: {val:.2f} at.%")

    if st.button("Quantify (with self-absorption)"):
        if 'result_noabs_norm' not in st.session_state:
            st.error("Please first quantify without self absorption.")
            st.stop()
        try:
            initial_composition = {el: val / 100 for el, val in st.session_state['result_noabs_norm'].items()}
        except Exception:
            st.error("Please first quantify without self absorption.")
            st.stop()
        result_abs = defaultdict(float)
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
                        if E > 0.1:
                            tau.append(xraylib.CS_Photo(Z, float(E)))
                        else:
                            tau.append(0.0)
                    except Exception:
                        tau.append(0.0)
                tau = np.array(tau)
                exc_int = np.trapz(valid_primary * tau, valid_energies)
                idx = (energies > E_line - 0.1) & (energies < E_line + 0.1)
                I = np.sum(counts[idx])
                mu_mix = compute_mu_mix(initial_composition, E_line)
                C_reabs = compute_self_absorption_correction(mu_mix, density, thickness_nm * 1e-7, angle_deg)
                corrected = (I / (omega * p * exc_int)) / C_reabs
                result_abs[el] += corrected
        total = sum(result_abs.values())
        result_abs_norm = {el: 100 * val / total for el, val in result_abs.items()}

        st.subheader("Atomic Percent (with self-absorption):")
        for el, val in result_abs_norm.items():
            st.write(f"{el}: {val:.2f} at.%")

        # Plot comparison
        st.subheader("Comparison of Quantification")
        elements_to_plot = sorted(set(st.session_state['result_noabs_norm']) | set(result_abs_norm))
        vals_noabs = [st.session_state['result_noabs_norm'].get(el, 0) for el in elements_to_plot]
        vals_abs = [result_abs_norm.get(el, 0) for el in elements_to_plot]

        fig2, ax2 = plt.subplots()
        x = np.arange(len(elements_to_plot))
        width = 0.35
        ax2.bar(x - width / 2, vals_noabs, width, label='No self-absorption')
        ax2.bar(x + width / 2, vals_abs, width, label='With self-absorption')
        ax2.set_ylabel('Atomic %')
        ax2.set_xticks(x)
        ax2.set_xticklabels(elements_to_plot)
        ax2.set_title('Comparison of Quantification Results')
        ax2.legend()
        st.pyplot(fig2)


