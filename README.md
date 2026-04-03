# Marketing Attribution & Budget Optimization Platform

This repository contains a full-stack marketing analytics platform built with Python, Scikit-learn, XGBoost, and Streamlit. It answers key marketing questions regarding channel ROIs, attribution models, and conversion lift, while also featuring an interactive budget allocation simulator.

## Features

1. **Synthetic Data Generation Pipelines (`src/dataset_generator.py`)** 
   - Dynamically models robust, user-level touchpoint sequences mimicking real multi-channel traffic across Paid Search (Google), Social (Meta), Email, and Organic Search.
2. **Multi-Touch Attribution Analysis (`src/attribution_models.py`)**
   - Compares performance metrics using standard marketing heuristics: First-Touch, Last-Touch, Linear, and Time-Decay attribution.
3. **Conversion Lift & Machine Learning Predictors (`src/predictive_models.py`)**
   - Trains an **XGBoost Classifier** to unpack complex, non-linear relationships and detect true feature lift/importance per advertising channel.
   - Evaluates a scalable **Logistic Regression** layer to model smooth elasticity of spend relative to conversions.
4. **Interactive Dashboard (`app.py`)**
   - A modern Streamlit front-end allowing users to visualize model attribution paths and utilize sliders to simulate future marketing budget shifts.

## Setup & Installation

Clone the repository and install the dependencies in a virtual environment:

```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Running the Platform

To view the dashboard, run the Streamlit server:

```bash
source venv/bin/activate
streamlit run app.py
```

The app will locally generate the marketing `synthetic_data.csv` on the first startup and automatically process the underlying models.

## Extending the Project
To ingest actual datasets such as the **Criteo Attribution Dataset** or Google BigQuery Analytics data instead of synthetic datasets, swap the initial ingestion point inside of `app.py` -> `load_data()`.
