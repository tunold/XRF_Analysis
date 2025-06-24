import streamlit as st
import matplotlib.pyplot as plt
from xrf_utils import (
    parse_spx_spectrum_with_metadata,
    quantify_xrf_custom_lines_weighted,
    quantify_with_self_absorption
)

st.set_page_config(page_title="XRF Quantification App", layout="wide")
st.title("X-ray Fluorescence (XRF) Quantification Tool")

# File upload
col1, col2 = st.columns(2)
with col1:
    primary_file = st.file_uploader("Upload primary spectrum (.spx)", type=["spx"], key="primary")
with col2:
    sample_file = st.file_uploader("Upload sample spectrum (.spx)", type=["spx"], key="sample")

# State for results
if "result_noabs" not in st.session_state:
    st.session_state.result_noabs = None

if primary_file and sample_file:
    # Load spectra
    energy_primary, primary_counts, meta_primary = parse_spx_spectrum_with_metadata(primary_file)
    energies, counts, meta_sample = parse_spx_spectrum_with_metadata(sample_file)

    st.subheader("Spectra")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(energy_primary, primary_counts, label="Primary")
    ax.plot(energies, counts, label="Sample")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("XRF Spectra")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Quantification Settings")
    elements = st.multiselect("Select elements to quantify", ["Cs", "Pb", "Sn", "I", "Br"], default=["Cs", "Pb", "Sn", "I", "Br"])
    line_options = ["Kα1", "Kβ1", "Lα1", "Lβ1"]

    line_selection = {}
    for el in elements:
        selected_lines = st.multiselect(f"Lines for {el}", ["Kα1", "Kβ1", "Lα1", "Lβ1"], default=["Lα1", "Lβ1"], key=f"line_{el}")
        if selected_lines:
            line_selection[el] = selected_lines

    density = st.number_input("Sample density [g/cm³]", value=5.0, key="density_input")
    thickness_nm = st.number_input("Film thickness [nm]", value=500.0, key="thickness_input")
    angle_deg = st.slider("Detector angle [deg]", 20, 90, 45, key="angle_input")

    col3, col4 = st.columns(2)

    with col3:
        if st.button("Quantify (without self-absorption)"):
            try:
                result_noabs, _ = quantify_xrf_custom_lines_weighted(
                    energies, counts, energy_primary, primary_counts, elements, line_selection
                )
                st.session_state.result_noabs = result_noabs

                st.subheader("Atomic Percent (no self-absorption):")
                for el, val in result_noabs.items():
                    st.write(f"{el}: {val:.2f} at.%")

            except Exception as e:
                st.error(f"Error during quantification without self-absorption: {e}")

    with col4:
        if st.button("Quantify (with self-absorption)"):
            if st.session_state.result_noabs is None:
                st.warning("Please quantify without self-absorption first.")
            else:
                try:
                    result_abs = quantify_with_self_absorption(
                        energies, counts, energy_primary, primary_counts,
                        elements, line_selection, st.session_state.result_noabs,
                        density, thickness_nm, angle_deg
                    )
                    st.subheader("Atomic Percent (with self-absorption):")
                    for el, val in result_abs.items():
                        st.write(f"{el}: {val:.2f} at.%")
                except Exception as e:
                    st.error(f"Error during self-absorption correction: {e}")