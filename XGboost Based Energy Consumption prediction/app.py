import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2ecc71;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to create time series features (same as notebook)
def create_features(df):
    """Create time series features from datetime index"""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    return df

# Load model and features
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        model = joblib.load('energy_consumption_v0.pkl')
        with open('features.json', 'r') as f:
            features = json.load(f)
        return model, features
    except FileNotFoundError:
        st.error("""
        ⚠️ Model file not found!
        
        Please run the notebook to train and save the model first:
        1. Open the notebook
        2. Run all cells to train the model
        3. Add the save code at the end
        4. Save the model as 'energy_consumption_model.pkl'
        """)
        return None, None

# Function to make predictions
def predict_consumption(model, df, features):
    """Make predictions using the trained model"""
    x = df[features]
    predictions = model.predict(x)
    return predictions

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}

# Function to plot predictions
def plot_predictions(df, days=30):
    plot_df = df.tail(days * 24).copy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['PJME_MW'],
        mode='lines', name='Actual',
        line=dict(color='#2ecc71', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['prediction'],
        mode='lines', name='Predicted',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Energy Consumption: Actual vs Predicted (Last {days} Days)',
        xaxis_title='Date', yaxis_title='Energy (MW)',
        hovermode='x unified', template='plotly_white', height=500
    )
    return fig

# Function to plot feature importance
def plot_feature_importance(model, features):
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title='Feature Importance', color='importance',
                 color_continuous_scale='Greens', height=400)
    fig.update_layout(xaxis_title='Importance Score', yaxis_title='Feature')
    return fig

# Function to predict for a specific date/time
def predict_single(model, date, features):
    """Predict consumption for a single datetime"""
    features_dict = {
        'hour': date.hour,
        'dayofweek': date.weekday(),
        'year': date.year,
        'month': date.month,
        'quarter': (date.month - 1) // 3 + 1,
        'dayofyear': date.timetuple().tm_yday
    }
    input_df = pd.DataFrame([features_dict])
    return model.predict(input_df)[0]

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/energy--v1.png", width=80)
        st.title("⚡ Energy Predictor")
        st.markdown("---")
        
        # File upload
        st.subheader("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file with 'Datetime' and 'PJME_MW' columns",
            type=['csv']
        )
        
        st.markdown("---")
        st.subheader("📊 About")
        st.info("""
        This app uses an XGBoost model trained on historical energy consumption data.
        The model uses time-based features:
        - Hour of day
        - Day of week
        - Month
        - Quarter
        - Day of year
        - Year
        """)
        
        # Load model
        model, features = load_model()
        
        if model is None:
            st.stop()
    
    # Main content
    st.markdown('<p class="main-header">⚡ Energy Consumption Predictor</p>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file)
            df = df.set_index('Datetime')
            df.index = pd.to_datetime(df.index)
            df = create_features(df)
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{df.index.min().date()} to {df.index.max().date()}")
        with col3:
            st.metric("Avg Consumption", f"{df['PJME_MW'].mean():,.0f} MW")
        with col4:
            st.metric("Max Consumption", f"{df['PJME_MW'].max():,.0f} MW")
        
        # Make predictions
        with st.spinner("Making predictions..."):
            df['prediction'] = predict_consumption(model, df, features)
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df[['PJME_MW', 'prediction']].head(100), use_container_width=True)
        
        # Split data (use same split as notebook: before/after 2015)
        train = df[df.index < '2015-01-01']
        test = df[df.index >= '2015-01-01']
        
        # Metrics
        st.markdown('<p class="sub-header">📈 Model Performance</p>', unsafe_allow_html=True)
        metrics = calculate_metrics(test['PJME_MW'], test['prediction'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['RMSE']:,.0f}</div>
                <div class="metric-label">RMSE (MW)</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['MAE']:,.0f}</div>
                <div class="metric-label">MAE (MW)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['R²']:.4f}</div>
                <div class="metric-label">R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['MAPE']:.2f}%</div>
                <div class="metric-label">MAPE</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown('<p class="sub-header">📊 Visualizations</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Predictions", "Feature Importance", "Pattern Analysis"])
        
        with tab1:
            days = st.slider("Number of days to display:", 7, 90, 30)
            fig = plot_predictions(test, days)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = plot_feature_importance(model, features)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                hourly_avg = df.groupby('hour')['PJME_MW'].mean().reset_index()
                fig = px.line(hourly_avg, x='hour', y='PJME_MW', 
                             title='Average by Hour', markers=True, height=400)
                fig.update_layout(xaxis_title='Hour', yaxis_title='MW')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                monthly_avg = df.groupby('month')['PJME_MW'].mean().reset_index()
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: months[x-1])
                fig = px.bar(monthly_avg, x='month_name', y='PJME_MW',
                            title='Average by Month', color='PJME_MW',
                            color_continuous_scale='Greens', height=400)
                fig.update_layout(xaxis_title='Month', yaxis_title='MW')
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction tool
        st.markdown('<p class="sub-header">🔮 Predict Future Consumption</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pred_date = st.date_input("Select Date", datetime.now().date())
            pred_hour = st.selectbox("Select Hour", range(24), format_func=lambda x: f"{x:02d}:00")
            
            if st.button("Predict", type="primary", use_container_width=True):
                pred_datetime = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=pred_hour)
                prediction = predict_single(model, pred_datetime, features)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div>Predicted Energy Consumption</div>
                    <div class="prediction-value">{prediction:,.0f} MW</div>
                    <div style="font-size: 0.9rem;">for {pred_datetime.strftime('%A, %B %d, %Y at %H:00')}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Consumption peaks: 7-9 AM & 5-8 PM
            - Weekdays > Weekends
            - Higher in winter & summer
            """)
        
        # Download results
        st.markdown("---")
        results_df = test[['PJME_MW', 'prediction']].copy()
        results_df['error'] = np.abs(results_df['PJME_MW'] - results_df['prediction'])
        results_df.index.name = 'Datetime'
        
        csv = results_df.to_csv()
        st.download_button("📥 Download Predictions (CSV)", csv, "predictions.csv", "text/csv")
        
    else:
        # No file uploaded
        st.markdown('<p class="main-header">⚡ Energy Consumption Predictor</p>', unsafe_allow_html=True)
        
        st.info("### 👈 Please upload your energy consumption data to get started!")
        
        st.markdown("""
        ### 📁 Expected Data Format
        
        | Datetime | PJME_MW |
        |----------|---------|
        | 2002-12-31 01:00:00 | 26498.0 |
        | 2002-12-31 02:00:00 | 25147.0 |
        
        **Requirements:**
        - Column names: `Datetime` and `PJME_MW`
        - Datetime format: YYYY-MM-DD HH:MM:SS
        
        ### 🚀 Features
        
        - Upload your data and get instant predictions
        - View model performance metrics
        - Interactive visualizations
        - Predict for any future date/time
        - Download results as CSV
        """)
        
        # Sample data
        st.markdown("---")
        st.subheader("📥 Try with Sample Data")
        
        dates = pd.date_range(start='2023-01-01', end='2023-01-07 23:00:00', freq='H')
        np.random.seed(42)
        consumption = 30000 + np.random.randn(len(dates)) * 5000 + \
                      np.sin(dates.hour * np.pi / 12) * 8000 + \
                      (dates.weekday < 5) * 3000
        
        sample_df = pd.DataFrame({'Datetime': dates, 'PJME_MW': consumption.round(1)})
        sample_csv = sample_df.to_csv(index=False)
        
        st.download_button("📥 Download Sample Data", sample_csv, "sample_data.csv", "text/csv")

if __name__ == "__main__":
    main()