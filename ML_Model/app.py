"""
Air Quality Index (AQI) Prediction and Monitoring System
A comprehensive Streamlit dashboard for AQI forecasting and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and model artifacts
@st.cache_data(ttl=3600)  # Cache for 1 hour to get fresh data
def load_data():
    """Load and combine air quality datasets from multiple sources including live data"""
    all_datasets = []
    
    # Historical datasets
    dataset_urls = [
        "https://raw.githubusercontent.com/harshchauhan01/Air-Quality-Analyzer/refs/heads/main/weather_data.csv",
        "https://raw.githubusercontent.com/harshchauhan01/Air-Quality-Analyzer/refs/heads/main/weather_data_v1.csv",
        # Add more dataset URLs here as needed
    ]
    
    # Government API for today's live data
    live_api_url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=579b464db66ec23bdd000001d2c2499fa4944ae76ca8d20687c6d3e7&format=csv&limit=4000"
    
    # Load historical datasets
    for idx, url in enumerate(dataset_urls, 1):
        try:
            df = pd.read_csv(url)
            df['data_source'] = f'historical_{idx}'
            all_datasets.append(df)
            # st.sidebar.success(f"✅ Loaded dataset {idx}: {len(df)} records")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load dataset {idx}: {str(e)}")
    
    # Load today's live data from Government API
    try:
        live_df = pd.read_csv(live_api_url)
        live_df['data_source'] = 'live_today'
        all_datasets.append(live_df)
        # st.sidebar.success(f"✅ Loaded live data: {len(live_df)} records")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load live data: {str(e)}")
    
    # Combine all datasets
    if not all_datasets:
        st.error("❌ Failed to load any datasets")
        return None
    
    try:
        # Concatenate all dataframes
        data = pd.concat(all_datasets, ignore_index=True)
        
        # Standardize date column
        if 'last_update' in data.columns:
            data['last_update'] = pd.to_datetime(data['last_update'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        
        # Remove duplicates based on key columns
        if all(col in data.columns for col in ['city', 'station', 'pollutant_id', 'last_update']):
            data = data.drop_duplicates(subset=['city', 'station', 'pollutant_id', 'last_update'], keep='last')
        
        # st.sidebar.info(f"📊 Total combined records: {len(data):,}")
        
        return data
    except Exception as e:
        st.error(f"Error combining datasets: {e}")
        return None

import os
@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing objects from the current folder only"""
    try:
        current_folder = os.getcwd()                   # Get current working directory
        file_path = os.path.join(current_folder, "model_artifacts.pkl")

        with open(file_path, 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts

    except FileNotFoundError:
        st.warning("model_artifacts.pkl not found in the current folder. "
                   "Please ensure the file exists.")
        return None

# AQI calculation functions
def calculate_aqi(pollutant, value):
    """Calculate AQI based on Indian National AQI standards"""
    if pd.isna(value) or value < 0:
        return np.nan
    
    # Official Indian AQI breakpoints (concentration ranges and corresponding AQI ranges)
    # Format: (Concentration_Low, Concentration_High, AQI_Low, AQI_High)
    aqi_breakpoints = {
        'PM2.5': [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), 
                  (90, 120, 201, 300), (120, 250, 301, 400), (250, 380, 401, 500)],
        'PM10': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), 
                 (250, 350, 201, 300), (350, 430, 301, 400), (430, 550, 401, 500)],
        'NO2': [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200), 
                (180, 280, 201, 300), (280, 400, 301, 400), (400, 800, 401, 500)],
        'SO2': [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200), 
                (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2100, 401, 500)],
        'CO': [(0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10, 101, 200), 
               (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)],
        'OZONE': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200), 
                  (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
        'NH3': [(0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200), 
                (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 2400, 401, 500)]
    }
    
    if pollutant not in aqi_breakpoints:
        return np.nan
    
    breakpoints = aqi_breakpoints[pollutant]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= value <= bp_hi:
            # Linear interpolation formula: I = [(I_hi - I_lo) / (C_hi - C_lo)] × (C - C_lo) + I_lo
            if bp_hi == bp_lo:
                aqi = aqi_lo
            else:
                aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (value - bp_lo) + aqi_lo
            return round(aqi)
    
    # If value exceeds all breakpoints, return maximum AQI
    return 500

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if pd.isna(aqi):
        return 'Unknown', '#gray', '⚪'
    elif aqi <= 50:
        return 'Good', '#00E400', '🟢'
    elif aqi <= 100:
        return 'Satisfactory', '#FFFF00', '🟡'
    elif aqi <= 200:
        return 'Moderate', '#FF7E00', '🟠'
    elif aqi <= 300:
        return 'Poor', '#FF0000', '🔴'
    elif aqi <= 400:
        return 'Very Poor', '#8F3F97', '🟣'
    else:
        return 'Severe', '#7E0023', '🟤'

def get_health_advisory(aqi):
    """Get health advisory based on AQI"""
    if aqi <= 50:
        return "Air quality is excellent. Perfect for outdoor activities!"
    elif aqi <= 100:
        return "Air quality is acceptable. Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi <= 200:
        return "Sensitive groups should reduce prolonged outdoor exertion. General public can continue normal activities."
    elif aqi <= 300:
        return "Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor activities."
    elif aqi <= 400:
        return "Everyone should avoid prolonged outdoor exertion. Sensitive groups should avoid ALL outdoor activities."
    else:
        return "EMERGENCY! Stay indoors. Avoid all outdoor activities. Health warning of emergency conditions."

# Main app
def main():
    # Sidebar
    st.sidebar.title("🌍 AQI Dashboard")
    st.sidebar.markdown("---")
    
    # Refresh data button
    if st.sidebar.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict AQI", "📊 Analytics", "🗺️ City Comparison", "📈 State Trends", "ℹ️ About"]
    )
    
    # Load data with spinner
    with st.spinner("🔄 Loading air quality data..."):
        data = load_data()
    
    if data is None:
        st.error("❌ Failed to load data. Please check your internet connection and try refreshing.")
        if st.button("🔄 Retry"):
            st.cache_data.clear()
            st.rerun()
        return
    
    # Calculate AQI for data if not present
    if 'AQI' not in data.columns:
        # Calculate sub-index for each pollutant
        data['AQI_sub'] = data.apply(lambda row: calculate_aqi(row['pollutant_id'], row['pollutant_avg']), axis=1)
        
        # Group by location and time to get maximum sub-index (overall AQI)
        # This is the correct method as per Indian AQI standards
        if all(col in data.columns for col in ['city', 'station', 'last_update']):
            data['AQI'] = data.groupby(['city', 'station', 'last_update'])['AQI_sub'].transform('max')
        else:
            # Fallback: use sub-index as AQI if grouping columns not available
            data['AQI'] = data['AQI_sub']
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")
    
    # Data source filter
    if 'data_source' in data.columns:
        data_sources = ['All'] + sorted(data['data_source'].unique().tolist())
        selected_source = st.sidebar.selectbox(
            "Data Source",
            options=data_sources,
            index=0
        )
        
        if selected_source != 'All':
            data = data[data['data_source'] == selected_source]
            st.sidebar.info(f"Filtered to: {selected_source}")
    
    # Date range filter
    date_range = None
    if data['last_update'].notna().any():
        min_date = data['last_update'].min().date()
        max_date = data['last_update'].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Pollutant filter
    pollutants = st.sidebar.multiselect(
        "Pollutants",
        options=data['pollutant_id'].unique().tolist(),
        default=data['pollutant_id'].unique().tolist()
    )
    
    # AQI category filter
    aqi_categories = st.sidebar.multiselect(
        "AQI Categories",
        options=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'],
        default=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    )
    
    # Apply filters
    filtered_data = data.copy()
    
    # Apply date range filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = filtered_data[
            (filtered_data['last_update'].dt.date >= start_date) & 
            (filtered_data['last_update'].dt.date <= end_date)
        ]
    
    # Apply pollutant filter
    if pollutants:
        filtered_data = filtered_data[filtered_data['pollutant_id'].isin(pollutants)]
    
    # Add category column for filtering
    filtered_data['AQI_Category'] = filtered_data['AQI'].apply(lambda x: get_aqi_category(x)[0])
    filtered_data = filtered_data[filtered_data['AQI_Category'].isin(aqi_categories)]
    
    # Check if filtered data is empty
    if len(filtered_data) == 0:
        st.warning("⚠️ No data available with current filters. Please adjust your filter settings.")
    
    # Export data option
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Data")
    
    if len(filtered_data) > 0:
        csv = filtered_data.to_csv(index=False)
        st.sidebar.download_button(
            label="⬇️ Download Filtered Data (CSV)",
            data=csv,
            file_name=f"aqi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width='stretch'
        )
        
        # Show data info
        st.sidebar.info(f"📊 Filtered records: {len(filtered_data):,}")
    else:
        st.sidebar.info("No data to export")
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home(filtered_data)
    elif page == "🔮 Predict AQI":
        show_prediction(data)
    elif page == "📊 Analytics":
        show_analytics(filtered_data)
    elif page == "🗺️ City Comparison":
        show_city_comparison(filtered_data)
    elif page == "📈 State Trends":
        show_state_trends(filtered_data)
    else:
        show_about()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if data['last_update'].notna().any():
            latest_update = data['last_update'].max()
            st.caption(f"📅 Latest Data: {latest_update.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.caption("📅 Latest Data: N/A")
    with col2:
        st.caption(f"📊 Total Records: {len(data):,}")
    with col3:
        st.caption("💡 Data updates hourly")

def show_home(data):
    """Home page with overview statistics"""
    st.title("🌍 Air Quality Index Monitoring System")
    st.markdown("### Real-time air quality monitoring and forecasting for Indian cities")
    
    # Today's data metrics
       
    # Check if we have data
    if len(data) == 0:
        st.warning("⚠️ No data available to display. Please adjust your filters or refresh the data.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_aqi = data['AQI'].mean()
        if pd.notna(avg_aqi):
            category, color, emoji = get_aqi_category(avg_aqi)
            st.metric("Average AQI", f"{avg_aqi:.1f}", delta=None)
            st.markdown(f"<p style='color: {color}; font-size: 20px;'>{emoji} {category}</p>", unsafe_allow_html=True)
        else:
            st.metric("Average AQI", "N/A")
    
    with col2:
        cities_count = data['city'].nunique()
        st.metric("Cities Monitored", f"{cities_count:,}")
        st.markdown("<p style='font-size: 14px; color: gray;'>Across India</p>", unsafe_allow_html=True)
    
    with col3:
        stations_count = data['station'].nunique()
        st.metric("Monitoring Stations", f"{stations_count:,}")
        st.markdown("<p style='font-size: 14px; color: gray;'>Active stations</p>", unsafe_allow_html=True)
    
    with col4:
        readings_count = len(data)
        st.metric("Total Readings", f"{readings_count:,}")
        st.markdown("<p style='font-size: 14px; color: gray;'>Data points</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top/Bottom cities
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Top 10 Cleanest Cities")
        cleanest = data.groupby('city')['AQI'].mean().sort_values().head(10)
        
        fig = go.Figure(go.Bar(
            x=cleanest.values,
            y=cleanest.index,
            orientation='h',
            marker=dict(color='lightgreen', line=dict(color='darkgreen', width=1))
        ))
        fig.update_layout(height=400, xaxis_title="Average AQI", yaxis_title="City")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("🔴 Top 10 Most Polluted Cities")
        polluted = data.groupby('city')['AQI'].mean().sort_values(ascending=False).head(10)
        
        fig = go.Figure(go.Bar(
            x=polluted.values,
            y=polluted.index,
            orientation='h',
            marker=dict(color='lightcoral', line=dict(color='darkred', width=1))
        ))
        fig.update_layout(height=400, xaxis_title="Average AQI", yaxis_title="City")
        st.plotly_chart(fig, width='stretch')
    
    # AQI distribution
    st.markdown("---")
    st.subheader("📊 AQI Category Distribution")
    
    category_data = data['AQI'].apply(lambda x: get_aqi_category(x)[0]).value_counts()
    colors_map = {'Good': '#00E400', 'Satisfactory': '#FFFF00', 'Moderate': '#FF7E00', 
                  'Poor': '#FF0000', 'Very Poor': '#8F3F97', 'Severe': '#7E0023'}
    
    fig = go.Figure(data=[go.Pie(
        labels=category_data.index,
        values=category_data.values,
        marker=dict(colors=[colors_map.get(cat, 'gray') for cat in category_data.index])
    )])
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')
    
    # Recent alerts
    st.markdown("---")
    st.subheader("⚠️ Cities Requiring Attention (AQI > 200)")
    
    alert_cities = data.groupby('city')['AQI'].mean()
    alert_cities = alert_cities[alert_cities > 200].sort_values(ascending=False)
    
    if len(alert_cities) > 0:
        for city, aqi in alert_cities.head(10).items():
            category, color, emoji = get_aqi_category(aqi)
            st.markdown(f"**{emoji} {city}**: AQI {aqi:.1f} - *{category}*")
    else:
        st.success("✅ All monitored cities have acceptable air quality!")

def show_prediction(data):
    """AQI prediction page"""
    st.title("🔮 AQI Prediction")
    st.markdown("### Predict Air Quality Index for any city")
    
    # Load model
    model_artifacts = load_model_artifacts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        # City and State selection
        col_a, col_b = st.columns(2)
        with col_a:
            cities = sorted(data['city'].unique().tolist())
            selected_city = st.selectbox("Select City", cities)
        
        with col_b:
            city_data = data[data['city'] == selected_city]
            state = city_data['state'].iloc[0] if len(city_data) > 0 else "Unknown"
            st.text_input("State", value=state, disabled=True)
        
        # Date and time
        col_c, col_d, col_e = st.columns(3)
        with col_c:
            pred_date = st.date_input("Date", value=datetime.now().date())
        with col_d:
            pred_hour = st.slider("Hour", 0, 23, 12)
        with col_e:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**{pred_hour:02d}:00**")
        
        st.markdown("---")
        st.subheader("Pollutant Levels")
        
        col1_p, col2_p, col3_p = st.columns(3)
        
        with col1_p:
            pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=999.0, value=50.0, step=1.0)
            pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=999.0, value=75.0, step=1.0)
        
        with col2_p:
            no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, max_value=999.0, value=40.0, step=1.0)
            so2 = st.number_input("SO2 (μg/m³)", min_value=0.0, max_value=9999.0, value=20.0, step=1.0)
        
        with col3_p:
            co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=999.0, value=1.0, step=0.1)
            ozone = st.number_input("Ozone (μg/m³)", min_value=0.0, max_value=9999.0, value=50.0, step=1.0)
        
        nh3 = st.number_input("NH3 (μg/m³)", min_value=0.0, max_value=9999.0, value=10.0, step=1.0)
        
        predict_btn = st.button("🔮 Predict AQI", type="primary", width='stretch')
    
    with col2:
        st.subheader("Quick Presets")
        
        if st.button("😊 Clean Air", width='stretch'):
            st.session_state.update({
                'pm25': 20.0, 'pm10': 35.0, 'no2': 15.0,
                'so2': 5.0, 'co': 0.5, 'ozone': 30.0, 'nh3': 5.0
            })
            st.rerun()
        
        if st.button("😐 Moderate Air", width='stretch'):
            st.session_state.update({
                'pm25': 70.0, 'pm10': 120.0, 'no2': 90.0,
                'so2': 100.0, 'co': 3.0, 'ozone': 110.0, 'nh3': 500.0
            })
            st.rerun()
        
        if st.button("😷 Polluted Air", width='stretch'):
            st.session_state.update({
                'pm25': 200.0, 'pm10': 300.0, 'no2': 250.0,
                'so2': 500.0, 'co': 12.0, 'ozone': 200.0, 'nh3': 1000.0
            })
            st.rerun()
        
        st.markdown("---")
        st.info("💡 **Tip**: Use presets for quick testing or enter custom pollutant values.")
    
    if predict_btn:
        pollutants = {
            'PM2.5': pm25, 'PM10': pm10, 'NO2': no2,
            'SO2': so2, 'CO': co, 'OZONE': ozone, 'NH3': nh3
        }
        
        # Calculate individual AQIs for reference
        individual_aqis = {}
        for pollutant, value in pollutants.items():
            aqi = calculate_aqi(pollutant, value)
            if not pd.isna(aqi):
                individual_aqis[pollutant] = aqi
        
        # Use ML model for prediction if available
        if model_artifacts is not None:
            try:
                # Extract model components
                model = model_artifacts['model']
                scaler = model_artifacts['scaler']
                le_city = model_artifacts['le_city']
                le_state = model_artifacts['le_state']
                feature_cols = model_artifacts['feature_cols']
                
                # Get city location data
                city_info = data[data['city'] == selected_city]
                if len(city_info) > 0:
                    latitude = city_info['latitude'].iloc[0]
                    longitude = city_info['longitude'].iloc[0]
                else:
                    latitude = 22.0
                    longitude = 78.0
                
                # Encode city and state
                try:
                    city_encoded = le_city.transform([selected_city])[0]
                except:
                    city_encoded = 0
                
                try:
                    state_encoded = le_state.transform([state])[0]
                except:
                    state_encoded = 0
                
                # Determine dominant pollutant
                max_pollutant = max(pollutants, key=pollutants.get)
                pollutant_avg_value = pollutants[max_pollutant]
                
                # Create feature vector
                feature_vector = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'year': pred_date.year,
                    'month': pred_date.month,
                    'day': pred_date.day,
                    'hour': pred_hour,
                    'dayofweek': pred_date.weekday(),
                    'is_weekend': 1 if pred_date.weekday() in [5, 6] else 0,
                    'city_encoded': city_encoded,
                    'state_encoded': state_encoded,
                    'pollutant_avg': pollutant_avg_value,
                    'station_id': 0,
                    'station_avg_pollutant': pollutant_avg_value,
                    'station_max_pollutant': pollutant_avg_value * 1.2,
                    'station_min_pollutant': pollutant_avg_value * 0.8
                }
                
                # Add one-hot encoded pollutant columns
                for pollutant in ['CO', 'NH3', 'NO2', 'OZONE', 'PM10', 'PM2.5', 'SO2']:
                    feature_vector[f'pollutant_{pollutant}'] = 1 if pollutant == max_pollutant else 0
                
                # Ensure all required features are present
                for col in feature_cols:
                    if col not in feature_vector:
                        feature_vector[col] = 0
                
                # Create DataFrame with correct column order
                X_pred = pd.DataFrame([feature_vector])[feature_cols]
                
                # Scale features
                X_pred_scaled = scaler.transform(X_pred)
                
                # Predict using ML model
                predicted_aqi = model.predict(X_pred_scaled)[0]
                predicted_aqi = max(0, predicted_aqi)  # Ensure non-negative
                
                st.success("✅ Using trained ML model for prediction")
                
            except Exception as e:
                st.warning(f"⚠️ Model prediction failed: {str(e)}. Using formula-based calculation.")
                predicted_aqi = max(individual_aqis.values()) if individual_aqis else 0
        else:
            # Fallback to formula-based calculation
            st.info("ℹ️ Using formula-based AQI calculation (model not loaded)")
            predicted_aqi = max(individual_aqis.values()) if individual_aqis else 0
        
        if predicted_aqi > 0:
            dominant_pollutant = max(individual_aqis, key=individual_aqis.get) if individual_aqis else 'PM2.5'
            
            category, color, emoji = get_aqi_category(predicted_aqi)
            advisory = get_health_advisory(predicted_aqi)
            
            st.markdown("---")
            st.markdown(f"## {emoji} Predicted AQI: {predicted_aqi:.0f}")
            st.markdown(f"<h3 style='color: {color};'>{category}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Dominant Pollutant:** {dominant_pollutant}")
            
            # Health advisory
            st.markdown("### 🏥 Health Advisory")
            
            if category in ['Poor', 'Very Poor', 'Severe']:
                st.error(advisory)
            elif category == 'Moderate':
                st.warning(advisory)
            else:
                st.success(advisory)
            
            # Individual pollutant AQIs
            st.markdown("### 📊 Individual Pollutant AQI Breakdown")
            
            cols = st.columns(len(individual_aqis))
            for idx, (pollutant, aqi) in enumerate(individual_aqis.items()):
                with cols[idx]:
                    cat, col, em = get_aqi_category(aqi)
                    st.metric(pollutant, f"{aqi:.0f}", delta=None)
                    st.markdown(f"<p style='color: {col};'>{em} {cat}</p>", unsafe_allow_html=True)
        else:
            st.error("Unable to calculate AQI. Please check pollutant values.")

def show_analytics(data):
    """Analytics page with detailed visualizations"""
    st.title("📊 Air Quality Analytics")
    
    # Check if data is available
    if len(data) == 0:
        st.warning("⚠️ No data available for analytics. Please adjust your filters.")
        return
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean AQI", f"{data['AQI'].mean():.1f}")
    with col2:
        st.metric("Median AQI", f"{data['AQI'].median():.1f}")
    with col3:
        st.metric("Max AQI", f"{data['AQI'].max():.1f}")
    with col4:
        st.metric("Std Dev", f"{data['AQI'].std():.1f}")
    
    st.markdown("---")
    
    # Temporal trends
    st.subheader("📅 Temporal Trends")
    
    if data['last_update'].notna().any():
        # Aggregate by date
        data['date'] = pd.to_datetime(data['last_update']).dt.date
        daily_aqi = data.groupby('date')['AQI'].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_aqi['date'],
            y=daily_aqi['AQI'],
            mode='lines+markers',
            name='Average AQI',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4)
        ))
        
        # Add threshold lines
        fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Satisfactory")
        fig.add_hline(y=200, line_dash="dash", line_color="orange", annotation_text="Moderate")
        fig.add_hline(y=300, line_dash="dash", line_color="red", annotation_text="Poor")
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="AQI",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
    
    # Hourly patterns
    st.markdown("---")
    st.subheader("⏰ Hourly AQI Patterns")
    
    data['hour'] = pd.to_datetime(data['last_update']).dt.hour
    hourly_aqi = data.groupby('hour')['AQI'].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_aqi['hour'],
        y=hourly_aqi['mean'],
        mode='lines+markers',
        name='Average',
        line=dict(color='steelblue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_aqi['hour'],
        y=hourly_aqi['mean'] + hourly_aqi['std'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_aqi['hour'],
        y=hourly_aqi['mean'] - hourly_aqi['std'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.2)',
        fill='tonexty',
        name='Std Dev'
    ))
    
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="AQI",
        height=400,
        xaxis=dict(tickmode='linear', dtick=2)
    )
    st.plotly_chart(fig, width='stretch')
    
    # Pollutant comparison
    st.markdown("---")
    st.subheader("🧪 Pollutant Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pollutant_avg = data.groupby('pollutant_id')['pollutant_avg'].mean().sort_values(ascending=False)
        
        fig = go.Figure(go.Bar(
            x=pollutant_avg.values,
            y=pollutant_avg.index,
            orientation='h',
            marker=dict(color='lightblue', line=dict(color='darkblue', width=1))
        ))
        fig.update_layout(
            title="Average Pollutant Concentrations",
            xaxis_title="Average Value",
            yaxis_title="Pollutant",
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        pollutant_count = data['pollutant_id'].value_counts()
        
        fig = go.Figure(go.Bar(
            x=pollutant_count.values,
            y=pollutant_count.index,
            orientation='h',
            marker=dict(color='lightcoral', line=dict(color='darkred', width=1))
        ))
        fig.update_layout(
            title="Number of Readings per Pollutant",
            xaxis_title="Count",
            yaxis_title="Pollutant",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

def show_city_comparison(data):
    """City comparison page"""
    st.title("🗺️ City-wise AQI Comparison")
    
    # Check if data is available
    if len(data) == 0:
        st.warning("⚠️ No data available for comparison. Please adjust your filters.")
        return
    
    # Multi-select for cities
    all_cities = sorted(data['city'].unique().tolist())
    
    if len(all_cities) == 0:
        st.warning("⚠️ No cities found in the data.")
        return
    
    selected_cities = st.multiselect(
        "Select cities to compare (max 10)",
        options=all_cities,
        default=all_cities[:5] if len(all_cities) >= 5 else all_cities,
        max_selections=10
    )
    
    if not selected_cities:
        st.warning("Please select at least one city.")
        return
    
    # Filter data
    city_data = data[data['city'].isin(selected_cities)]
    
    # Summary metrics
    st.subheader("📊 Summary Statistics")
    
    city_stats = city_data.groupby('city')['AQI'].agg(['mean', 'min', 'max', 'std']).round(2)
    city_stats['Category'] = city_stats['mean'].apply(lambda x: get_aqi_category(x)[0])
    city_stats.columns = ['Avg AQI', 'Min AQI', 'Max AQI', 'Std Dev', 'Category']
    
    st.dataframe(city_stats, width='stretch')
    
    # Comparison chart
    st.markdown("---")
    st.subheader("📊 Average AQI Comparison")
    
    city_avg = city_data.groupby('city')['AQI'].mean().sort_values(ascending=False)
    colors = [get_aqi_category(aqi)[1] for aqi in city_avg.values]
    
    fig = go.Figure(go.Bar(
        x=city_avg.values,
        y=city_avg.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color='black', width=1)),
        text=[f"{val:.1f}" for val in city_avg.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        xaxis_title="Average AQI",
        yaxis_title="City",
        height=max(400, len(selected_cities) * 50)
    )
    st.plotly_chart(fig, width='stretch')
    
    # Time series comparison
    if data['last_update'].notna().any():
        st.markdown("---")
        st.subheader("📈 AQI Trends Over Time")
        
        city_data['date'] = pd.to_datetime(city_data['last_update']).dt.date
        time_series = city_data.groupby(['date', 'city'])['AQI'].mean().reset_index()
        
        fig = px.line(
            time_series,
            x='date',
            y='AQI',
            color='city',
            title="Daily Average AQI by City"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    # Pollutant breakdown
    st.markdown("---")
    st.subheader("🧪 Pollutant Breakdown by City")
    
    pollutant_city = city_data.groupby(['city', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
    
    fig = px.bar(
        pollutant_city,
        x='city',
        y='pollutant_avg',
        color='pollutant_id',
        title="Average Pollutant Levels",
        barmode='group'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')

def show_state_trends(data):
    """State-wise trends page"""
    st.title("📈 State-wise AQI Trends")
    
    # Check if data is available
    if len(data) == 0:
        st.warning("⚠️ No data available for state analysis. Please adjust your filters.")
        return
    
    # State selector
    all_states = sorted(data['state'].unique().tolist())
    
    if len(all_states) == 0:
        st.warning("⚠️ No states found in the data.")
        return
    
    selected_state = st.selectbox("Select State", all_states)
    
    state_data = data[data['state'] == selected_state]
    
    # State overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_aqi = state_data['AQI'].mean()
        category, color, emoji = get_aqi_category(avg_aqi)
        st.metric("Average AQI", f"{avg_aqi:.1f}")
        st.markdown(f"<p style='color: {color};'>{emoji} {category}</p>", unsafe_allow_html=True)
    
    with col2:
        cities_count = state_data['city'].nunique()
        st.metric("Cities", f"{cities_count}")
    
    with col3:
        stations_count = state_data['station'].nunique()
        st.metric("Stations", f"{stations_count}")
    
    with col4:
        readings_count = len(state_data)
        st.metric("Readings", f"{readings_count:,}")
    
    # City rankings in state
    st.markdown("---")
    st.subheader(f"🏙️ Cities in {selected_state}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 Cleanest Cities**")
        cleanest = state_data.groupby('city')['AQI'].mean().sort_values().head(10)
        for idx, (city, aqi) in enumerate(cleanest.items(), 1):
            cat, col, em = get_aqi_category(aqi)
            st.markdown(f"{idx}. {em} **{city}**: {aqi:.1f}")
    
    with col2:
        st.markdown("**🔴 Most Polluted Cities**")
        polluted = state_data.groupby('city')['AQI'].mean().sort_values(ascending=False).head(10)
        for idx, (city, aqi) in enumerate(polluted.items(), 1):
            cat, col, em = get_aqi_category(aqi)
            st.markdown(f"{idx}. {em} **{city}**: {aqi:.1f}")
    
    # Temporal trends
    st.markdown("---")
    st.subheader("📈 AQI Trends Over Time")
    
    if state_data['last_update'].notna().any():
        state_data['date'] = pd.to_datetime(state_data['last_update']).dt.date
        daily_trend = state_data.groupby('date')['AQI'].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_trend['date'],
            y=daily_trend['AQI'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='royalblue', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average AQI",
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    # Pollutant distribution
    st.markdown("---")
    st.subheader("🧪 Pollutant Distribution")
    
    pollutant_data = state_data.groupby('pollutant_id')['pollutant_avg'].describe()
    
    fig = go.Figure()
    for pollutant in state_data['pollutant_id'].unique():
        pollutant_values = state_data[state_data['pollutant_id'] == pollutant]['pollutant_avg']
        fig.add_trace(go.Box(y=pollutant_values, name=pollutant))
    
    fig.update_layout(
        yaxis_title="Concentration",
        height=400
    )
    st.plotly_chart(fig, width='stretch')
    
    # Heatmap of cities
    st.markdown("---")
    st.subheader("🗺️ City AQI Heatmap")
    
    city_aqi = state_data.groupby('city')['AQI'].mean().sort_values(ascending=False).head(20)
    
    fig = go.Figure(go.Bar(
        x=city_aqi.values,
        y=city_aqi.index,
        orientation='h',
        marker=dict(
            color=city_aqi.values,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="AQI")
        )
    ))
    
    fig.update_layout(
        xaxis_title="Average AQI",
        yaxis_title="City",
        height=max(400, len(city_aqi) * 30)
    )
    st.plotly_chart(fig, width='stretch')

def show_about():
    """About page"""
    st.title("ℹ️ About the AQI Prediction System")
    
    st.markdown("""
    ## 🌍 Air Quality Index (AQI) Monitoring System
    
    This comprehensive dashboard provides real-time air quality monitoring and forecasting 
    for Indian cities based on the **Indian National AQI standards**.
    
    ### 📊 Features
    
    - **Real-time Monitoring**: Track AQI across multiple cities and states
    - **Predictive Analytics**: Forecast future AQI levels
    - **Health Advisories**: Get personalized health recommendations
    - **Interactive Visualizations**: Explore trends and patterns
    - **Multi-city Comparison**: Compare air quality across locations
    - **State-level Analysis**: Deep dive into state-specific trends
    
    ### 🎯 AQI Categories (Indian Standards)
    
    | AQI Range | Category | Health Impact | Color |
    |-----------|----------|---------------|-------|
    | 0-50 | Good | Minimal impact | 🟢 Green |
    | 51-100 | Satisfactory | Minor breathing discomfort to sensitive people | 🟡 Yellow |
    | 101-200 | Moderate | Breathing discomfort to people with lung disease | 🟠 Orange |
    | 201-300 | Poor | Breathing discomfort to most people | 🔴 Red |
    | 301-400 | Very Poor | Respiratory illness on prolonged exposure | 🟣 Purple |
    | 401-500 | Severe | Affects healthy people and seriously impacts those with existing diseases | 🟤 Brown |
    
    ### 🧪 Monitored Pollutants
    
    - **PM2.5**: Fine particulate matter (≤2.5 micrometers)
    - **PM10**: Particulate matter (≤10 micrometers)
    - **NO2**: Nitrogen Dioxide
    - **SO2**: Sulfur Dioxide
    - **CO**: Carbon Monoxide
    - **O3**: Ozone
    - **NH3**: Ammonia
    
    ### 📈 Machine Learning Models
    
    The system uses multiple ML models for prediction:
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LSTM (Deep Learning)
    
    ### 📚 Data Sources
    
    The system combines multiple data sources for comprehensive coverage:
    
    1. **Historical Datasets**: Past air quality records from monitoring stations
       - `weather_data.csv` - Primary historical dataset
       - `weather_data_v1.csv` - Extended historical data
       - Additional datasets as they become available
    
    2. **Live Government API**: Real-time data from Government of India
       - Source: [data.gov.in](https://data.gov.in)
       - Updates: Fetched hourly for latest readings
       - Coverage: 4000+ recent measurements across India
    
    The data is automatically combined, deduplicated, and refreshed every hour to provide 
    the most up-to-date air quality information.
    
    ### 👨‍💻 Technical Stack
    
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **ML**: Scikit-learn, XGBoost, TensorFlow
    - **Data Processing**: Pandas, NumPy
    
    ---
    
    💡 **Tip**: Use the sidebar filters to customize your view and explore specific regions or time periods.
    
    ⚠️ **Disclaimer**: This system is for informational purposes. For official air quality data, 
    please refer to government sources like CPCB (Central Pollution Control Board).
    """)

if __name__ == "__main__":
    main()
