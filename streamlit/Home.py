import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Loan Analysis Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4169E1;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: medium;
        color: #333333;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">Loan Analysis Dashboard</p>', unsafe_allow_html=True)

# Introduction
st.markdown('<p class="sub-header">Welcome to the Loan Analysis Platform</p>', unsafe_allow_html=True)

st.write("""
This interactive dashboard provides comprehensive analysis and visualization of loan data.
Navigate through the pages to explore different aspects of the dataset and gain valuable insights.
""")

# Main page content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Available Pages")
    st.write("""
    - **Loan Dataset**: View and explore the loan dataset
    - **EDA & Visualization**: Comprehensive exploratory data analysis with interactive visualizations
    """)

with col2:
    st.subheader("Quick Navigation")
    if st.button("Go to Loan Dataset"):
        st.switch_page("pages/Loan_Dataset.py")
    if st.button("Go to EDA & Visualization"):
        st.switch_page("pages/EDA_Visualization.py")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit") 