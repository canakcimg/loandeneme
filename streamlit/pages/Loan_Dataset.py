import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(
    page_title="Loan Dataset",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("Loan Dataset Explorer")
st.write("This page allows you to explore and understand the loan dataset.")

# Function to load data
@st.cache_data
def load_data():
    try:
        # Check if data exists in the data directory
        if os.path.exists("data/loan_data.csv"):
            df = pd.read_csv("data/loan_data.csv")
        else:
            # Sample data if file doesn't exist
            st.warning("Sample data is being used. Upload your own data for analysis.")
            # Create sample loan data
            np.random.seed(42)
            n = 1000
            
            data = {
                'loan_id': [f'LOAN{i:06d}' for i in range(1, n+1)],
                'loan_amount': np.random.randint(1000, 100000, n),
                'term': np.random.choice(['36 months', '60 months'], n),
                'interest_rate': np.round(np.random.uniform(5.0, 25.0, n), 2),
                'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n),
                'employment_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', 
                                                     '6 years', '7 years', '8 years', '9 years', '10+ years'], n),
                'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], n),
                'annual_income': np.round(np.random.normal(70000, 30000, n), 2),
                'loan_status': np.random.choice(['Current', 'Fully Paid', 'Late (31-120 days)', 'Default', 'Charged Off'], n,
                                              p=[0.6, 0.2, 0.1, 0.05, 0.05])
            }
            
            df = pd.DataFrame(data)
            
            # Save the data
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/loan_data.csv", index=False)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

# Data upload option
st.subheader("Upload Your Own Loan Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data successfully loaded!")
        # Save the uploaded data
        df.to_csv("data/loan_data.csv", index=False)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display data information
if not df.empty:
    st.subheader("Dataset Overview")
    
    # Basic dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Number of Features", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # Display data sample
    with st.expander("Show Data Sample"):
        st.dataframe(df.head(10))
    
    # Display column information
    with st.expander("Column Information"):
        # Create a DataFrame with column info
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%',
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info)
    
    # Data filtering
    st.subheader("Filter Data")
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter by categorical values
    if categorical_cols:
        st.write("Filter by categorical values:")
        cat_filters = {}
        
        # Create up to 3 columns for categorical filters
        cols = st.columns(min(3, len(categorical_cols)))
        
        for i, col_name in enumerate(categorical_cols[:3]):  # Limit to 3 categorical filters
            with cols[i % 3]:
                unique_values = ['All'] + list(df[col_name].dropna().unique())
                cat_filters[col_name] = st.selectbox(f"Select {col_name}", unique_values)
        
        # Apply categorical filters
        filtered_df = df.copy()
        for col, value in cat_filters.items():
            if value != 'All':
                filtered_df = filtered_df[filtered_df[col] == value]
    
    # Filter by numeric range
    if numeric_cols:
        st.write("Filter by numeric range:")
        num_filters = {}
        
        # Create columns for numeric filters
        cols = st.columns(min(2, len(numeric_cols)))
        
        for i, col_name in enumerate(numeric_cols[:2]):  # Limit to 2 numeric filters for simplicity
            with cols[i % 2]:
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                num_filters[col_name] = st.slider(
                    f"{col_name} range",
                    min_val,
                    max_val,
                    (min_val, max_val)
                )
        
        # Apply numeric filters
        for col, (min_val, max_val) in num_filters.items():
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.write(f"Showing {len(filtered_df)} of {len(df)} records")
    st.dataframe(filtered_df)
    
    # Download filtered data
    if st.button("Download Filtered Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_loan_data.csv",
            mime="text/csv"
        )
else:
    st.error("No data available. Please upload a dataset.")

# Footer
st.markdown("---")
st.info("Navigate to the EDA & Visualization page for detailed analysis.") 