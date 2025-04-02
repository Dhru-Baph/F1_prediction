import fastf1
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st # type: ignore

import os
import fastf1

# Ensure the cache directory exists
cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable FastF1 cache
fastf1.Cache.enable_cache(cache_dir)


fastf1.Cache.enable_cache('cache')

def fetch_f1_data(year, round_number):
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
        results = results.rename(columns={'FullName': 'Driver'})
        
        for col in ['Q1', 'Q2', 'Q3']:
            results[col + '_sec'] = results[col].apply(
                lambda x: x.total_seconds() if pd.notnull(x) else None
            )
        
        return results
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def convert_time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    try:
        if ':' in time_str:
            minutes, seconds = time_str.split(':')
            return float(minutes) * 60 + float(seconds)
        else:
            return float(time_str)
    except (ValueError, TypeError) as e:
        st.warning(f"Warning: Could not convert time: {time_str}, Error: {e}")
        return None

def clean_data(df):
    df['Q1_sec'] = df['Q1'].apply(convert_time_to_seconds)
    df['Q2_sec'] = df['Q2'].apply(convert_time_to_seconds)
    df['Q3_sec'] = df['Q3'].apply(convert_time_to_seconds)
    return df.dropna()  # removes NaN cases

def visualize_data(df):
    sns.boxplot(data=df[['Q1_sec', 'Q2_sec', 'Q3_sec']])
    plt.title('Qualifying Lap Times (seconds)')
    plt.ylabel('Lap Time (seconds)')
    st.pyplot()

def apply_performance_factors(predictions_df):
    base_time = 89.5  # in seconds
    
    team_factors = {
        'Red Bull Racing': 0.997,    # -0.3s from base
        'Ferrari': 0.998,          # -0.2s from base
        'McLaren': 0.999,          # -0.15s from base
        'Mercedes': 0.999,         # -0.15s from base
        'Aston Martin': 1.001,     # +0.1s from base
        'RB': 1.002,              # +0.2s from base
        'Williams': 1.003,         # +0.3s from base
        'Haas F1 Team': 1.004,     # +0.4s from base
        'Kick Sauber': 1.004,      # +0.4s from base (Audi development)
        'Alpine': 1.005,           # +0.5s from base
    }
    
    driver_factors = {
        'Max Verstappen': 0.998,     # -0.2s (exceptional)
        'Charles Leclerc': 0.999,    # -0.1s (very strong qualifier)
        'Carlos Sainz': 0.999,       # -0.1s (very consistent)
        'Lando Norris': 0.999,       # -0.1s (McLaren leader)
        'Oscar Piastri': 1.000,      # Base time (strong)
        'Sergio Perez': 1.000,       # Base time
        'Lewis Hamilton': 1.000,     # Base time
        'George Russell': 1.000,     # Base time
        'Fernando Alonso': 1.000,    # Base time
        'Lance Stroll': 1.001,       # +0.1s
        'Alex Albon': 1.001,         # +0.1s
        'Daniel Ricciardo': 1.001,   # +0.1s
        'Yuki Tsunoda': 1.002,       # +0.2s
        'Valtteri Bottas': 1.002,    # +0.2s
        'Zhou Guanyu': 1.003,        # +0.3s
        'Kevin Magnussen': 1.003,    # +0.3s
        'Nico Hulkenberg': 1.003,    # +0.3s
        'Logan Sargeant': 1.004,     # +0.4s
        'Pierre Gasly': 1.004,       # +0.4s
        'Esteban Ocon': 1.004,       # +0.4s
    }

    for idx, row in predictions_df.iterrows():
        team_factor = team_factors.get(row['Team'], 1.005)
        driver_factor = driver_factors.get(row['Driver'], 1.002)
        
        base_prediction = base_time * team_factor * driver_factor
        
        random_variation = np.random.uniform(-0.1, 0.1)
        predictions_df.loc[idx, 'Predicted_Q3'] = base_prediction + random_variation
    
    return predictions_df

def display_predictions_table(results_df):
    # Create a DataFrame with the required columns
    prediction_table = results_df[['Position', 'Driver', 'Team', 'Predicted_Q3']].reset_index(drop=True)
    
    # Display the table in Streamlit
    st.table(prediction_table)

import requests

def get_wikipedia_driver_image(driver_name):
    """Fetch the driver's image URL from Wikipedia API."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{driver_name.replace(' ', '_')}"
    response = requests.get(url).json()
    
    if 'thumbnail' in response:
        return response['thumbnail']['source']
    else:
        return "Image not found"


import numpy as np
import pandas as pd
import streamlit as st

def predict_gp(model, latest_data):
    driver_teams = {
        'Max Verstappen': 'Red Bull Racing',
        'Sergio Perez': 'Red Bull Racing',
        'Charles Leclerc': 'Ferrari',
        'Carlos Sainz': 'Ferrari',
        'Lewis Hamilton': 'Mercedes',
        'George Russell': 'Mercedes',
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Daniel Ricciardo': 'RB',
        'Yuki Tsunoda': 'RB',
        'Alexander Albon': 'Williams',
        'Logan Sargeant': 'Williams',
        'Valtteri Bottas': 'Kick Sauber',
        'Zhou Guanyu': 'Kick Sauber',
        'Kevin Magnussen': 'Haas F1 Team',
        'Nico Hulkenberg': 'Haas F1 Team',
        'Pierre Gasly': 'Alpine',
        'Esteban Ocon': 'Alpine'
    }

    # Create DataFrame with Drivers & Teams
    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])

    # Check if latest_data is valid
    if latest_data is None or latest_data.empty:
        st.error("No valid data available for prediction.")
        return

    # Merge driver times with team assignments
    results_df = results_df.merge(latest_data, on='Driver', how='left')

    # Convert 'Q1_sec' and 'Q2_sec' to seconds if they are in timedelta format
    for col in ['Q1_sec', 'Q2_sec']:
        if col in results_df.columns:
            if np.issubdtype(results_df[col].dtype, np.timedelta64):
                results_df[col] = results_df[col].dt.total_seconds()
            results_df[col].fillna(results_df[col].median(), inplace=True)  # Fill NaNs with median

    # Prepare data for prediction
    X_predict = results_df[['Q1_sec', 'Q2_sec']].copy()

    # Predict Q3 times using trained model
    try:
        results_df['Predicted_Q3'] = model.predict(X_predict)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return

    # Sort drivers by predicted Q3 times
    results_df = results_df.sort_values(by='Predicted_Q3')

    # Assign positions
    results_df['Position'] = range(1, len(results_df) + 1)

    # Display Top Driver Image
    # Get the top 3 drivers and their images
    # Get the top 3 drivers and their images
    top_drivers = results_df.iloc[:3]['Driver']
    image_urls = [get_wikipedia_driver_image(driver) for driver in top_drivers]
    
    # Create three columns for the images
    col1, col2, col3 = st.columns(3)
    
    # Define a fixed height for uniformity
    fixed_height = 300  # Adjust as needed
    
    # Display images in each column
    for col, image_url, driver in zip([col1, col2, col3], image_urls, top_drivers):
        with col:
            if image_url:
                st.markdown(
                    f"""
                    <div style="text-align: center; height: {fixed_height}px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                        <img src="{image_url}" width="200" 
                             style="border-radius: 15px; border: 3px solid white; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); max-height: 200px; object-fit: cover;">
                        <p style="font-weight: bold; margin-top: 10px;">{driver}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    # Display the prediction table below the images
    display_predictions_table(results_df)




def main():
    st.title("Formula 1 Predictor")
    st.markdown("### Let's Predict!")

    # Apply background gradient
    st.markdown(
        """
        <style>
            /* Apply background gradient */
            body {
                background: linear-gradient(to right, #000000, #FF1801);
                color: white;
            }
            
            /* Streamlit main content area */
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(to right, #000000, #FF1801);
            }
            
            /* Sidebar styling (if needed) */
            [data-testid="stSidebar"] {
                background: linear-gradient(to bottom, #FF1801, #000000);
            }
            
            /* Modify text color */
            h1, h2, h3, h4, h5, h6, p, div {
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
        )
    

    current_year = st.selectbox("Select the year you are predicting for:", options=[2023, 2024, 2025], index=2)
    schedule = fastf1.get_event_schedule(current_year)
    race_mapping = dict(zip(schedule['EventName'], schedule['RoundNumber'])) # Map race name to round number

    # Select race based on fetched schedule
    race_name = st.selectbox("Select the race track:", options=list(race_mapping.keys()))
    # Get the round number based on the selected race
    race_num = race_mapping[race_name]
    
    # Fetch previous two years' schedules
    previous_years = [current_year - 1, current_year - 2]
    previous_race_nums = {}
    
    for year in previous_years:
        try:
            schedule_prev = fastf1.get_event_schedule(year)
            race_mapping_prev = dict(zip(schedule_prev['EventName'], schedule_prev['RoundNumber']))
            previous_race_nums[year] = race_mapping_prev.get(race_name, None)  # Handle missing race mappings
        except:
            previous_race_nums[year] = None  # In case of error fetching schedule
    
    if st.button("Predict Q3 Times"):
        all_data = []
    
        # Fetch past race data for training
        for round_num in range(1, race_num):
            st.write(f"Fetching data for {current_year} round {round_num}...")
            df = fetch_f1_data(current_year, round_num)
            if df is not None:
                df['Year'] = current_year
                df['Round'] = round_num
                all_data.append(df)
    
        # Fetch data from the previous two years
        for year, round_num in previous_race_nums.items():
            if round_num is not None:
                st.write(f"Fetching data for {year} round {round_num}...")
                prev_year_data = fetch_f1_data(year, round_num)
                if prev_year_data is not None:
                    prev_year_data['Year'] = year
                    prev_year_data['Round'] = round_num
                    all_data.append(prev_year_data)
    
        # Train the model only if there is valid data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='all')

        if not valid_data.empty:
            # Ensure 'Driver' column is available
            if "Driver" not in valid_data.columns:
                st.error("❌ 'Driver' column missing in historical data.")
                return
            
            # Data preprocessing
            imputer = SimpleImputer(strategy='median')
            
            X = valid_data[['Q1_sec', 'Q2_sec']]
            y = valid_data['Q3_sec'].values.reshape(-1, 1)  # Ensure correct shape for imputation
            
            # Impute missing values
            X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            y_clean = pd.Series(imputer.fit_transform(y).ravel(), name='Q3_sec')
        
            # Train model
            model = LinearRegression()
            model.fit(X_clean, y_clean)
        
            # Fetch latest race data for prediction (if available)
            latest_data = fetch_f1_data(current_year, race_num)

            # Ensure 'valid_data' is passed correctly
            predict_gp(model, valid_data.copy(), latest_data)
        
            # Evaluate Model Performance
            y_pred = model.predict(X_clean)
            mae = mean_absolute_error(y_clean, y_pred)
            r2 = r2_score(y_clean, y_pred)
        
            # Display metrics in a well-formatted way
            st.markdown("## 📊 Model Performance Metrics")
            st.success(f"✅ **Mean Absolute Error:** `{mae:.2f}` seconds")
            st.success(f"✅ **R² Score:** `{r2:.2f}`")
        else:
            st.error("❌ No valid training data available.")
    else:
        st.error("❌ No stored historical F1 data found.")


if __name__ == "__main__":
    main()
