import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from xrf_utils import (
    parse_spx_spectrum_with_metadata,
    quantify_xrf_custom_lines_weighted,
    quantify_xrf_with_self_absorption,
)

st.set_page_config(layout="wide")
st.title("XRF Quantification with Optional Self-Absorption Correction")

# --- File Upload ---
col1, col2 = st.columns(2)

with col1:
    primary_file = st.file_uploader("Upload primary spectrum (.spx)", type="spx", key="primary")
with col2:
    sample_file = st.file_uploader("Upload sample spectrum (.spx)", type="spx", key="sample")

# --- Parameters ---
st.sidebar.header("Analysis Parameters")
available_elements = ["Cs", "Sn", "Pb", "I", "Br"]
selected_elements = st.sidebar.multiselect("Select elements", available_elements, default=available_elements)

line_options = ["Kα1", "Kβ1", "Lα1", "Lβ1"]
line_selection = {}
for el in selected_elements:
    lines = st.sidebar.multiselect(f"{el} lines", line_options, default=["Lα1", "Lβ1"], key=el)
    line_selection[el] = lines

density = st.sidebar.number_input("Sample density [g/cm³]", value=5.0, key="density_input")
thickness_nm = st.sidebar.number_input("Sample thickness [nm]", value=500.0, key="thickness_input")
angle_deg = st.sidebar.number_input("Detector angle [deg]", value=45.0, key="angle_input")

# --- Load and Display Spectra ---
if primary_file and sample_file:
    energy_primary, primary_counts, meta1 = parse_spx_spectrum_with_metadata(primary_file)
    energies, counts, meta2 = parse_spx_spectrum_with_metadata(sample_file)

    fig, ax = plt.subplots()
    ax.plot(energy_primary, primary_counts, label="Primary Spectrum", alpha=0.6)
    ax.plot(energies, counts, label="Sample Spectrum", alpha=0.6)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Spectra")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Perform Quantification ---
    col1, col2 = st.columns(2)
    if "result_noabs_norm" not in st.session_state:
        st.session_state.result_noabs_norm = None
    result_abs_norm = None

    with col1:
        if st.button("Quantify (No Self-Absorption)"):
            try:
                st.session_state.result_noabs_norm, st.session_state.detail_noabs  = quantify_xrf_custom_lines_weighted(
                    energies, counts, energy_primary, primary_counts,
                    selected_elements, line_selection
                )
                st.subheader("Quantification Result (No Self-Absorption)")
                for el, val in result_noabs_norm.items():
                    st.write(f"{el}: {val:.2f} at.%")
            except Exception as e:
                st.error(f"Error during quantification without self-absorption: {e}")

    with col2:
        if st.button("Quantify (With Self-Absorption)"):
            if st.session_state.result_noabs_norm is None:
                st.error("Please first quantify without self absorption.")

            else:
                try:
                    result_abs_norm, detail_abs = quantify_xrf_with_self_absorption(
                        energies, counts, energy_primary, primary_counts,
                        selected_elements, line_selection,
                        result_noabs_norm,
                        density=density,
                        thickness_cm=thickness_nm * 1e-7,
                        angle_deg=angle_deg
                    )
                    st.subheader("Quantification Result (With Self-Absorption)")
                    for el, val in result_abs_norm.items():
                        st.write(f"{el}: {val:.2f} at.%")
                except Exception as e:
                    st.error(f"Error during quantification with self-absorption: {e}")

    # --- Comparison Table ---
    if st.session_state.get("result_noabs_norm") and st.session_state.get("result_abs_norm"):

        st.subheader("Comparison Table")
        data = {
            "Element": [],
            "Lines": [],
            "No Self-Abs. (at.%)": [],
            "With Self-Abs. (at.%)": []
        }
        for el in selected_elements:
            data["Element"].append(el)
            data["Lines"].append(", ".join(line_selection.get(el, [])))
            data["No Self-Abs. (at.%)"].append(result_noabs_norm.get(el, 0.0))
            data["With Self-Abs. (at.%)"].append(result_abs_norm.get(el, 0.0))

        import pandas as pd
        st.table(pd.DataFrame(data))
