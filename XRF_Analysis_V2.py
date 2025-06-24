# xrf_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from xrf_utils import parse_spx_spectrum_with_metadata, quantify_without_self_absorption, quantify_with_self_absorption
from xrf_utils import get_line_code_and_shell, compute_mu_mix, compute_self_absorption_correction
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
    ax.plot(energy_primary, primary_counts, label="Primary spectrum")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    ax.set_title("XRF Spectra")
    ax.set_yscale('log')
    ax.set_ylim(1,max(counts))
    ax.legend()
    st.pyplot(fig)

    elements = st.multiselect("Select elements", ["Cs", "Pb", "Sn", "I", "Br"], default=["Cs", "Pb", "Sn", "I", "Br"])
    available_lines = ["Kα1", "Kβ1", "Lα1", "Lβ1"]

    line_selection_default = {
        "Cs": ["Lβ1"],
        "Sn": ["Lα1", "Lβ1"],
        "Pb": ["Lα1", "Lβ1"],
        "I":  ["Lα1"],
        "Br": ["Kα1", "Kβ1"]
    }

    line_selection = {}
    for el in elements:
        selected_lines = st.multiselect(f"Select lines for {el}", available_lines, default=["Lα1"] if el in ["Sn", "Pb", "I"] else ["Kα1"], key=f"line_{el}")
        line_selection[el] = selected_lines

    density = st.number_input("Sample density [g/cm³]", value=5.0, key="density_input")
    thickness_nm = st.number_input("Film thickness [nm]", value=500.0, key="thickness_input")
    angle_deg = st.number_input("Detector angle [deg]", value=45.0, key="angle_input")

    if st.button("Quantify (no self-absorption)"):
        try:
            # Limit primary spectrum to safe energy range for xraylib
            energy_primary = energy_primary[(energy_primary > 1.0) & (energy_primary < 100.0)]
            primary_counts = primary_counts[:len(energy_primary)]

            result_noabs_norm = quantify_without_self_absorption(energies, counts, energy_primary, primary_counts,
                                                                 elements, line_selection)
            st.session_state['result_noabs_norm'] = result_noabs_norm

            st.subheader("Atomic Percent (no self-absorption):")
            for el, val in st.session_state['result_noabs_norm'].items():
                st.write(f"{el}: {val:.2f} at.%")
        except Exception as e:
            st.error(f"Error during quantification without self-absorption: {e}")

    if st.button("Quantify (with self-absorption)"):
        if 'result_noabs_norm' not in st.session_state:
            st.error("Please first quantify without self absorption.")
            st.stop()

        try:
            # Limit primary spectrum to safe energy range for xraylib
            energy_primary = energy_primary[(energy_primary > 1.0) & (energy_primary < 100.0)]
            primary_counts = primary_counts[:len(energy_primary)]

            initial_composition = {el: val / 100 for el, val in st.session_state['result_noabs_norm'].items()}
            result_abs_norm = quantify_with_self_absorption(
                energies, counts, energy_primary, primary_counts,
                elements, line_selection, initial_composition,
                density, thickness_nm, angle_deg
            )

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

        except Exception as e:
            st.error(f"Error during quantification with self-absorption: {e}")
