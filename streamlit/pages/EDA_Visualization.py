import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="EDA & Visualization",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.chart-container {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("Exploratory Data Analysis & Visualization")
st.write("This page provides comprehensive exploratory data analysis with interactive visualizations.")

# Function to load data
@st.cache_data
def load_data():
    try:
        if os.path.exists("data/loan_data.csv"):
            df = pd.read_csv("data/loan_data.csv")
            return df
        else:
            st.error("No data found. Please go to the Loan Dataset page first to upload or generate sample data.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

if not df.empty:
    # Data preprocessing for visualization
    # Convert data types if necessary
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Sidebar for visualization options
    st.sidebar.title("Visualization Options")
    
    # Select visualization type
    viz_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Overview", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Custom Visualization"]
    )
    
    if viz_type == "Overview":
        st.header("Data Overview")
        
        # Dashboard metrics
        col1, col2, col3 = st.columns(3)
        
        if 'loan_amount' in df.columns:
            with col1:
                st.metric("Average Loan Amount", f"${df['loan_amount'].mean():,.2f}")
        
        if 'interest_rate' in df.columns:
            with col2:
                st.metric("Average Interest Rate", f"{df['interest_rate'].mean():.2f}%")
        
        if 'loan_status' in df.columns and 'Fully Paid' in df['loan_status'].unique():
            with col3:
                fully_paid_pct = (df['loan_status'] == 'Fully Paid').mean() * 100
                st.metric("Fully Paid Loans", f"{fully_paid_pct:.2f}%")
        
        # Data distribution overview
        st.subheader("Data Distribution Overview")
        
        if numeric_cols:
            # Create a histogram for each numeric column
            fig = make_subplots(rows=len(numeric_cols[:4]), cols=1, subplot_titles=numeric_cols[:4])
            
            for i, col in enumerate(numeric_cols[:4]):  # Limit to first 4 numeric columns
                fig.add_trace(
                    go.Histogram(x=df[col], name=col),
                    row=i+1, col=1
                )
            
            fig.update_layout(height=300 * min(len(numeric_cols[:4]), 4), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        if categorical_cols:
            # Create bar charts for categorical columns
            st.subheader("Categorical Data Overview")
            
            # Select columns to display
            display_cols = categorical_cols[:3]  # Limit to first 3 categorical columns
            
            for col in display_cols:
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'Count']
                fig = px.bar(
                    value_counts,
                    x=col,
                    y='Count',
                    title=f"Distribution of {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Univariate Analysis":
        st.header("Univariate Analysis")
        
        # Choose variable to analyze
        if numeric_cols:
            st.subheader("Numeric Variables")
            num_var = st.selectbox("Select a numeric variable", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    df, 
                    x=num_var,
                    title=f"Distribution of {num_var}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Descriptive statistics
                stats = df[num_var].describe().reset_index()
                stats.columns = ['Statistic', 'Value']
                st.dataframe(stats, use_container_width=True)
                
                # Box plot
                fig = px.box(df, y=num_var, title=f"Box Plot of {num_var}")
                st.plotly_chart(fig, use_container_width=True)
        
        if categorical_cols:
            st.subheader("Categorical Variables")
            cat_var = st.selectbox("Select a categorical variable", categorical_cols)
            
            # Bar chart
            value_counts = df[cat_var].value_counts().reset_index()
            value_counts.columns = [cat_var, 'Count']
            
            # Add percentage column
            value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    value_counts,
                    x=cat_var,
                    y='Count',
                    title=f"Distribution of {cat_var}",
                    text='Percentage',
                    color=cat_var
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(value_counts, use_container_width=True)
                
                # Pie chart
                fig = px.pie(
                    value_counts,
                    values='Count',
                    names=cat_var,
                    title=f"Proportion of {cat_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bivariate Analysis":
        st.header("Bivariate Analysis")
        
        # Select variables for analysis
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Select X variable", df.columns)
        
        with col2:
            y_var = st.selectbox("Select Y variable", [col for col in df.columns if col != x_var])
        
        # Determine the types of the selected variables
        x_type = "numeric" if df[x_var].dtype in ['int64', 'float64'] else "categorical"
        y_type = "numeric" if df[y_var].dtype in ['int64', 'float64'] else "categorical"
        
        # Create appropriate visualization based on variable types
        if x_type == "numeric" and y_type == "numeric":
            # Scatter plot for numeric vs numeric
            st.subheader(f"Scatter Plot: {x_var} vs {y_var}")
            
            fig = px.scatter(
                df,
                x=x_var,
                y=y_var,
                title=f"{x_var} vs {y_var}",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation information
            corr = df[[x_var, y_var]].corr().iloc[0, 1]
            st.info(f"Correlation coefficient: {corr:.4f}")
            
            # Hexbin plot for large datasets
            if len(df) > 1000:
                st.subheader("Hexbin Density Plot")
                fig = px.density_heatmap(
                    df,
                    x=x_var,
                    y=y_var,
                    title=f"Density Heatmap: {x_var} vs {y_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif x_type == "categorical" and y_type == "numeric":
            # Box plot for categorical vs numeric
            st.subheader(f"Box Plot: {y_var} by {x_var}")
            
            fig = px.box(
                df,
                x=x_var,
                y=y_var,
                title=f"{y_var} by {x_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart with mean values
            st.subheader(f"Mean {y_var} by {x_var}")
            
            agg_df = df.groupby(x_var)[y_var].mean().reset_index()
            fig = px.bar(
                agg_df,
                x=x_var,
                y=y_var,
                title=f"Mean {y_var} by {x_var}",
                color=x_var
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif x_type == "numeric" and y_type == "categorical":
            # Box plot for categorical vs numeric (flipped)
            st.subheader(f"Box Plot: {x_var} by {y_var}")
            
            fig = px.box(
                df,
                x=x_var,
                y=y_var,
                title=f"{x_var} by {y_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Histogram by category
            st.subheader(f"Distribution of {x_var} by {y_var}")
            
            fig = px.histogram(
                df,
                x=x_var,
                color=y_var,
                marginal="box",
                title=f"Distribution of {x_var} by {y_var}",
                barmode="overlay"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Both categorical
            # Heatmap for categorical vs categorical
            st.subheader(f"Contingency Table: {x_var} vs {y_var}")
            
            # Create contingency table
            contingency = pd.crosstab(df[x_var], df[y_var], normalize='all')
            
            # Plot heatmap
            fig = px.imshow(
                contingency,
                labels=dict(x=y_var, y=x_var, color="Proportion"),
                x=contingency.columns,
                y=contingency.index,
                title=f"Heatmap of {x_var} vs {y_var}"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Stacked bar chart
            st.subheader(f"Stacked Bar Chart: {x_var} vs {y_var}")
            
            # Create stacked bar chart
            contingency_count = pd.crosstab(df[x_var], df[y_var])
            fig = px.bar(
                contingency_count,
                barmode="stack",
                title=f"Stacked Bar Chart of {x_var} vs {y_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Multivariate Analysis":
        st.header("Multivariate Analysis")
        
        # Option for correlation analysis
        if len(numeric_cols) > 2:
            st.subheader("Correlation Matrix")
            
            # Generate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Plot heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix of Numeric Variables"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Select highly correlated pairs
            threshold = 0.5
            high_corr = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        high_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                st.subheader(f"Highly Correlated Variables (|r| > {threshold})")
                high_corr_df = pd.DataFrame(high_corr)
                st.dataframe(high_corr_df.sort_values(by='Correlation', key=abs, ascending=False), use_container_width=True)
            else:
                st.info(f"No variable pairs with correlation greater than {threshold} found.")
        
        # 3D scatter plot
        if len(numeric_cols) >= 3:
            st.subheader("3D Scatter Plot")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                x_var = st.selectbox("X Variable", numeric_cols, index=0)
            
            with col2:
                y_var = st.selectbox("Y Variable", [col for col in numeric_cols if col != x_var], index=0)
            
            with col3:
                z_var = st.selectbox("Z Variable", [col for col in numeric_cols if col not in [x_var, y_var]], index=0)
            
            with col4:
                color_var = st.selectbox("Color Variable", ['None'] + [col for col in df.columns if col not in [x_var, y_var, z_var]])
            
            # Create 3D scatter plot
            if color_var == 'None':
                fig = px.scatter_3d(
                    df,
                    x=x_var,
                    y=y_var,
                    z=z_var,
                    title=f"3D Scatter Plot: {x_var} vs {y_var} vs {z_var}"
                )
            else:
                fig = px.scatter_3d(
                    df,
                    x=x_var,
                    y=y_var,
                    z=z_var,
                    color=color_var,
                    title=f"3D Scatter Plot: {x_var} vs {y_var} vs {z_var}, colored by {color_var}"
                )
            
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Custom Visualization":
        st.header("Custom Visualization")
        
        # Advanced options for custom visualization
        chart_type = st.sidebar.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Violin Plot", "Heatmap", "Pie Chart", "Histogram"]
        )
        
        # Common parameters for all charts
        x_var = st.sidebar.selectbox("X-axis Variable", df.columns)
        
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Violin Plot"]:
            y_var = st.sidebar.selectbox("Y-axis Variable", [col for col in df.columns if col != x_var])
        
        color_var = st.sidebar.selectbox("Color Variable (optional)", ['None'] + [col for col in df.columns if col not in [x_var, y_var if 'y_var' in locals() else '']])
        color_var = None if color_var == 'None' else color_var
        
        # Render the selected chart type
        if chart_type == "Bar Chart":
            if df[x_var].dtype in ['int64', 'float64'] and len(df[x_var].unique()) > 20:
                st.warning(f"{x_var} has too many unique values for a bar chart. Consider using a histogram instead.")
            else:
                title = st.sidebar.text_input("Chart Title", f"{y_var} by {x_var}")
                
                if df[x_var].dtype in ['int64', 'float64'] and df[y_var].dtype in ['object', 'category']:
                    # Swap x and y for better visualization
                    fig = px.bar(df, x=y_var, y=x_var, color=color_var, title=title)
                else:
                    fig = px.bar(df, x=x_var, y=y_var, color=color_var, title=title)
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Line Chart":
            title = st.sidebar.text_input("Chart Title", f"{y_var} over {x_var}")
            
            if df[x_var].dtype in ['int64', 'float64'] or df[x_var].dtype == 'datetime64[ns]':
                # Group by x_var if it's not datetime and has many values
                if df[x_var].dtype != 'datetime64[ns]' and len(df[x_var].unique()) > 30:
                    st.info(f"Grouping data by {x_var} for better visualization")
                    
                    # Create bins for x_var
                    df['bin'] = pd.cut(df[x_var], bins=20)
                    
                    # Group by bin and calculate mean of y_var
                    grouped = df.groupby('bin')[y_var].mean().reset_index()
                    grouped[x_var] = grouped['bin'].apply(lambda x: x.mid)
                    
                    fig = px.line(grouped, x=x_var, y=y_var, title=title)
                else:
                    # Sort by x_var for proper line chart
                    sorted_df = df.sort_values(by=x_var)
                    fig = px.line(sorted_df, x=x_var, y=y_var, color=color_var, title=title)
            else:
                # For categorical x_var, group and calculate mean of y_var
                grouped = df.groupby(x_var)[y_var].mean().reset_index()
                fig = px.line(grouped, x=x_var, y=y_var, title=title, markers=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Scatter Plot":
            title = st.sidebar.text_input("Chart Title", f"{y_var} vs {x_var}")
            
            add_trendline = st.sidebar.checkbox("Add Trendline")
            
            if add_trendline:
                fig = px.scatter(df, x=x_var, y=y_var, color=color_var, title=title, trendline="ols")
            else:
                fig = px.scatter(df, x=x_var, y=y_var, color=color_var, title=title)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Box Plot":
            title = st.sidebar.text_input("Chart Title", f"Distribution of {y_var} by {x_var}")
            
            fig = px.box(df, x=x_var, y=y_var, color=color_var, title=title)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Violin Plot":
            title = st.sidebar.text_input("Chart Title", f"Distribution of {y_var} by {x_var}")
            
            fig = px.violin(df, x=x_var, y=y_var, color=color_var, title=title, box=True)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Heatmap":
            st.subheader("Heatmap")
            
            # For heatmap, we need two categorical variables
            if df[x_var].dtype in ['object', 'category'] or len(df[x_var].unique()) <= 20:
                y_var = st.sidebar.selectbox("Y-axis Variable", [col for col in df.columns if col != x_var and (df[col].dtype in ['object', 'category'] or len(df[col].unique()) <= 20)])
                
                if df[y_var].dtype in ['object', 'category'] or len(df[y_var].unique()) <= 20:
                    # Choose value to aggregate
                    value_var = st.sidebar.selectbox("Value to Aggregate", [None] + numeric_cols)
                    
                    title = st.sidebar.text_input("Chart Title", f"Heatmap of {x_var} vs {y_var}")
                    
                    if value_var:
                        # Create pivot table for heatmap
                        pivot = df.pivot_table(index=y_var, columns=x_var, values=value_var, aggfunc='mean')
                        fig = px.imshow(pivot, title=f"Mean {value_var} by {x_var} and {y_var}")
                    else:
                        # Create contingency table
                        contingency = pd.crosstab(df[y_var], df[x_var])
                        fig = px.imshow(contingency, title=f"Count of {x_var} and {y_var}")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"{y_var} has too many unique values for a heatmap. Please select another variable.")
            else:
                st.warning(f"{x_var} has too many unique values for a heatmap. Please select another variable.")
        
        elif chart_type == "Pie Chart":
            if df[x_var].dtype in ['object', 'category'] or len(df[x_var].unique()) <= 20:
                title = st.sidebar.text_input("Chart Title", f"Distribution of {x_var}")
                
                # Choose value to aggregate if available
                if numeric_cols:
                    value_var = st.sidebar.selectbox("Value to Sum (optional)", ['Count'] + numeric_cols)
                    
                    if value_var == 'Count':
                        # Use count as the value
                        value_counts = df[x_var].value_counts().reset_index()
                        value_counts.columns = [x_var, 'Count']
                        fig = px.pie(value_counts, values='Count', names=x_var, title=title)
                    else:
                        # Sum the selected numeric variable
                        agg_df = df.groupby(x_var)[value_var].sum().reset_index()
                        fig = px.pie(agg_df, values=value_var, names=x_var, title=f"Sum of {value_var} by {x_var}")
                else:
                    # Use count as the value
                    value_counts = df[x_var].value_counts().reset_index()
                    value_counts.columns = [x_var, 'Count']
                    fig = px.pie(value_counts, values='Count', names=x_var, title=title)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"{x_var} has too many unique values for a pie chart. Please select another variable.")
        
        elif chart_type == "Histogram":
            title = st.sidebar.text_input("Chart Title", f"Distribution of {x_var}")
            
            n_bins = st.sidebar.slider("Number of Bins", min_value=5, max_value=100, value=20)
            
            fig = px.histogram(df, x=x_var, color=color_var, nbins=n_bins, title=title, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("No data available. Please go to the Loan Dataset page first to upload or generate sample data.")

# Footer
st.markdown("---")
st.info("This tool provides extensive visualization capabilities for loan data analysis.") 