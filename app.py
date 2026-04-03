import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from src.attribution_models import prep_user_paths, calculate_heuristic_models
from src.predictive_models import BudgetSimulator

# Page config
st.set_page_config(page_title="Marketing Optimization Platform", layout="wide")

st.title("📈 Marketing Attribution & Budget Optimization Platform")

# 1. Load Data
@st.cache_data
def load_data():
    file_path = "synthetic_data.csv"
    if not os.path.exists(file_path):
        from src.dataset_generator import generate_synthetic_data
        generate_synthetic_data()
    
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    paths_df = pd.DataFrame(prep_user_paths(df))
    return df, paths_df

df, paths_df = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["Attribution Modeling", "Conversion Lift Analysis", "Budget Allocation Simulator"])

# --- TAB 1: Attribution Modeling ---
with tab1:
    st.header("Multi-Touch Attribution Analysis")
    st.markdown("Compare how different heuristic models credit conversions to marketing channels.")
    
    with st.spinner("Calculating attribution models..."):
        attr_df = calculate_heuristic_models(paths_df)
    
    # Melt for plotly
    attr_melt = attr_df.melt(id_vars="Channel", var_name="Model", value_name="Conversions")
    
    fig = px.bar(attr_melt, x="Channel", y="Conversions", color="Model", barmode="group",
                 title="Conversions by Channel & Attribution Model",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Conversion Lift Analysis (XGBoost) ---
with tab2:
    st.header("Conversion Lift & Feature Importance")
    st.markdown("Uses XGBoost to analyze the true lift and importance of each channel on driving conversions.")
    
    with st.spinner("Training predictive models..."):
        simulator = BudgetSimulator()
        metrics = simulator.train(paths_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance (AUC)")
        st.metric("XGBoost AUC", f"{metrics['xgb_auc']:.3f}", help="Score > 0.5 means the model predicts better than random.")
        st.metric("Logistic Regression AUC", f"{metrics['lr_auc']:.3f}")
        
    with col2:
        st.subheader("Channel Lift (XGBoost Feature Importance)")
        importance_df = pd.DataFrame({
            'Channel': list(metrics['feature_importance'].keys()),
            'Importance': list(metrics['feature_importance'].values())
        }).sort_values('Importance', ascending=True)
        
        fig2 = px.bar(importance_df, x='Importance', y='Channel', orientation='h',
                      title="Relative Importance of Channels",
                      color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: Budget Allocation Simulator ---
with tab3:
    st.header("Interactive Budget Simulator")
    st.markdown("Slide the budgets to see projected changes in revenue/conversions using the trained machine learning model.")
    
    if 'simulator' not in locals():
        simulator = BudgetSimulator()
        simulator.train(paths_df)
        
    baseline_convs = simulator.baseline_conversions
    
    st.write(f"**Current Baseline Conversions:** {baseline_convs:.0f}")
    
    st.markdown("### Adjust Budgets (%)")
    
    cols = st.columns(4)
    budget_sliders = {}
    
    channels = simulator.channels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (col, channel) in enumerate(zip(cols, channels)):
        with col:
             # Default 100%
             val = st.slider(f"{channel}", min_value=0, max_value=200, value=100, step=5, format="%d%%")
             budget_sliders[channel] = val / 100.0
             
    # Calculate simulation
    sim_results = simulator.simulate_budget(budget_sliders)
    
    st.markdown("---")
    st.subheader("Projected Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Projected Total Conversions", 
                  f"{sim_results['projected_conversions']:.0f}", 
                  f"{sim_results['delta_conversions']:.0f} ({sim_results['percent_change']:.1f}%)")
                  
    with res_col2:
        # Mini bar chart comparing before vs after
        fig_sim = go.Figure(data=[
            go.Bar(name='Baseline', x=['Conversions'], y=[baseline_convs]),
            go.Bar(name='Projected', x=['Conversions'], y=[sim_results['projected_conversions']])
        ])
        fig_sim.update_layout(title="Baseline vs. Projected", barmode='group')
        st.plotly_chart(fig_sim, use_container_width=True)
