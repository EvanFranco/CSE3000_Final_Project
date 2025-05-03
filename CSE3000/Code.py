import pandas as pd
import requests
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import folium
from folium.plugins import HeatMap
import datetime
import os

def fetch_nyc_crime_data():
    """Fetch crime data from NYC Open Data API"""
    try:
        url = "https://data.cityofnewyork.us/resource/5uac-w243.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Print the columns we got from the API
            print("\nColumns received from API:")
            print(df.columns.tolist())
            
            # Print sample of the data
            print("\nSample of the first few rows:")
            print(df.head())
            
            return df
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise

def preprocess_data(df):
    """Clean and prepare the data"""
    features = [
        'addr_pct_cd',     # Precinct
        'boro_nm',         # Borough
        'latitude',        # Location data
        'longitude',    
        'law_cat_cd',      # Crime category (target variable)
        'vic_age_group',   # Victim age group
        'vic_sex',         # Victim sex
        'ofns_desc'        # Offense description
    ]
    
    # Print available columns for debugging
    print("Available columns in dataset:", df.columns.tolist())
    
    # Verify and handle missing columns
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Remove missing columns from features list
        features = [col for col in features if col not in missing_cols]
    
    # Drop rows with missing values
    df = df.dropna(subset=features)
    
    # Convert latitude and longitude to float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Create synthetic time features
    print("Creating synthetic time features...")
    # Generate random hours for demonstration (in a real scenario, you'd use actual time data)
    np.random.seed(42)  # For reproducibility
    df['hour_of_day'] = np.random.randint(0, 24, size=len(df))
    df['is_night'] = (df['hour_of_day'] >= 20) | (df['hour_of_day'] <= 5)
    df['is_weekend'] = np.random.choice([True, False], size=len(df), p=[0.3, 0.7])
    
    return df[features + ['hour_of_day', 'is_night', 'is_weekend']]

def create_crime_heatmap(df):
    """Create and save an interactive heat map with crime type information"""
    try:
        # Ensure we have the required columns
        required_columns = ['latitude', 'longitude', 'ofns_desc', 'boro_nm']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove rows with missing values in required columns
        df = df.dropna(subset=required_columns)
        
        # Convert crime descriptions to string and clean them
        df['ofns_desc'] = df['ofns_desc'].astype(str).str.strip().str.upper()
        
        # Create base map centered on NYC
        nyc_map = folium.Map(location=[40.7128, -74.0060], 
                            zoom_start=11,
                            tiles='CartoDB positron')
        
        # Create a feature group for each crime type
        crime_groups = {}
        
        # Get unique crime types
        crime_types = df['ofns_desc'].unique()
        
        # Create a color map for different crime types
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                'darkpurple', 'pink', 'lightblue', 'lightgreen',
                'gray', 'black', 'lightgray']
        
        # Create a color dictionary for crime types
        color_dict = {}
        for i, crime_type in enumerate(crime_types):
            color_dict[crime_type] = colors[i % len(colors)]
        
        # Print data info for debugging
        print("\nDataset Info:")
        print(f"Total number of incidents: {len(df)}")
        print(f"Number of unique crime types: {len(crime_types)}")
        print("\nSample of crime types:", list(crime_types)[:5])
        
        # Add markers for each crime
        for idx, row in df.iterrows():
            try:
                crime_type = row['ofns_desc']
                lat, lon = float(row['latitude']), float(row['longitude'])
                
                # Skip if coordinates are invalid
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                
                # Create feature group for this crime type if it doesn't exist
                if crime_type not in crime_groups:
                    crime_groups[crime_type] = folium.FeatureGroup(name=crime_type)
                
                # Add a circle marker for this crime
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    popup=f"Crime: {crime_type}<br>Borough: {row['boro_nm']}",
                    color=color_dict[crime_type],
                    fill=True,
                    fillOpacity=0.7
                ).add_to(crime_groups[crime_type])
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        # Add all feature groups to the map
        for crime_type, group in crime_groups.items():
            group.add_to(nyc_map)
        
        # Add layer control to toggle crime types
        folium.LayerControl().add_to(nyc_map)
        
        # Save map to HTML file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'crime_maps/current_crime_map_{timestamp}.html'
        
        # Create directory if it doesn't exist
        os.makedirs('crime_maps', exist_ok=True)
        
        nyc_map.save(filename)
        print(f"\nCurrent crime map saved as: {filename}")
        
        # Print summary of crime types
        print("\nCrime Types Summary:")
        crime_summary = df['ofns_desc'].value_counts()
        for crime_type, count in crime_summary.items():
            print(f"{crime_type}: {count} incidents")
            
    except Exception as e:
        print(f"Error creating crime map: {str(e)}")
        print("DataFrame columns:", df.columns.tolist())
        print("\nSample of data:")
        print(df.head())

def train_model(df):
    """Train and evaluate the Random Forest model"""
    # Prepare features and target
    X = df.drop(['law_cat_cd', 'latitude', 'longitude'], axis=1)
    y = df['law_cat_cd']
    
    # Encode categorical variables
    le = LabelEncoder()
    for column in X.select_dtypes(include=['object']):
        X[column] = le.fit_transform(X[column])
        # Save the encoder for each column
        joblib.dump(le, f'le_{column}.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf.predict(X_test)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf

def predict_crime_times(df):
    """Analyze and predict crime patterns based on time"""
    try:
        print("\nStarting crime time prediction analysis...")
        
        # Create time-based features
        print("Creating time-based features...")
        if 'hour_of_day' not in df.columns:
            print("Error: Time features not found in dataset")
            print("Available columns:", df.columns.tolist())
            return None
            
        # Analyze crime patterns by hour
        print("\nAnalyzing crime patterns by hour...")
        hourly_patterns = df.groupby('hour_of_day')['ofns_desc'].count()
        print(hourly_patterns)
        
        # Find most common crime times
        most_common_hours = hourly_patterns.nlargest(5)
        print("\nMost Common Crime Hours:")
        for hour, count in most_common_hours.items():
            print(f"Hour {hour}: {count} crimes")
        
        # Analyze crime types by time of day
        print("\nAnalyzing crime types by time of day...")
        crime_by_time = df.groupby(['hour_of_day', 'ofns_desc']).size().unstack().fillna(0)
        print(crime_by_time)
        
        # Create a prediction model for crime times
        print("\nCreating time prediction model...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare features for time prediction
        time_features = ['hour_of_day', 'is_night', 'is_weekend', 'boro_nm']
        X_time = pd.get_dummies(df[time_features])
        
        # Encode crime types
        le = LabelEncoder()
        y_time = le.fit_transform(df['ofns_desc'])
        
        print(f"Training data shape: {X_time.shape}")
        print(f"Target data shape: {y_time.shape}")
        print(f"Number of unique crime types: {len(le.classes_)}")
        
        # Train the model
        time_model = RandomForestClassifier(n_estimators=100, random_state=42)
        time_model.fit(X_time, y_time)
        
        print("Model trained successfully")
        
        # Save the time prediction model and label encoder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('models', exist_ok=True)
        model_path = f'models/crime_time_predictor_{timestamp}.pkl'
        encoder_path = f'models/crime_label_encoder_{timestamp}.pkl'
        
        joblib.dump(time_model, model_path)
        joblib.dump(le, encoder_path)
        print(f"\nTime prediction model saved as: {model_path}")
        print(f"Label encoder saved as: {encoder_path}")
        
        return time_model, le
            
    except Exception as e:
        print(f"Error in crime time prediction: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return None, None

def create_predictive_heatmap(df, time_model, label_encoder):
    """Create a predictive map showing individual crime predictions at different times"""
    try:
        print("\nStarting predictive crime map creation...")
        
        # Validate input data
        if time_model is None or label_encoder is None:
            print("Error: Time model or label encoder is not available")
            return
        
        # Print available boroughs for debugging
        print("\nAvailable boroughs in dataset:", df['boro_nm'].unique())
        
        # Create a grid of points covering NYC
        nyc_bounds = {
            'min_lat': 40.4774,
            'max_lat': 40.9176,
            'min_lon': -74.2591,
            'max_lon': -73.7004
        }
        
        print("Creating grid points...")
        # Create grid points
        lat_points = np.linspace(nyc_bounds['min_lat'], nyc_bounds['max_lat'], 30)
        lon_points = np.linspace(nyc_bounds['min_lon'], nyc_bounds['max_lon'], 30)
        
        # Create base map
        print("Creating base map...")
        nyc_map = folium.Map(location=[40.7128, -74.0060], 
                           zoom_start=11,
                           tiles='CartoDB positron')
        
        # Create time layers for different hours
        hours = [0, 6, 12, 18]  # Midnight, 6 AM, Noon, 6 PM
        time_names = ['Midnight', '6 AM', 'Noon', '6 PM']
        time_colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c']  # Black, Blue, Orange, Green
        
        print("Generating predictions for each time period...")
        
        for hour, time_name, color in zip(hours, time_names, time_colors):
            print(f"\nProcessing {time_name}...")
            # Create feature group for this time
            time_group = folium.FeatureGroup(name=f'Crime Predictions - {time_name}')
            
            # Prepare features for prediction
            test_points = []
            total_points = len(lat_points) * len(lon_points)
            point_count = 0
            
            for lat in lat_points:
                for lon in lon_points:
                    point_count += 1
                    if point_count % 10 == 0:
                        print(f"Processed {point_count}/{total_points} points...")
                    
                    try:
                        # Find the nearest borough
                        distances = ((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)
                        nearest_idx = distances.idxmin()
                        nearest_borough = df.loc[nearest_idx, 'boro_nm']
                        
                        # Create features for prediction
                        features = pd.DataFrame({
                            'hour_of_day': [hour],
                            'is_night': [hour >= 20 or hour <= 5],
                            'is_weekend': [False],  # Default to weekday
                            'boro_nm': [nearest_borough]
                        })
                        
                        # Get prediction
                        features_dummies = pd.get_dummies(features)
                        # Ensure all required columns are present
                        missing_cols = set(time_model.feature_names_in_) - set(features_dummies.columns)
                        for col in missing_cols:
                            features_dummies[col] = 0
                        features_dummies = features_dummies[time_model.feature_names_in_]
                        
                        # Get prediction probabilities
                        probabilities = time_model.predict_proba(features_dummies)[0]
                        max_prob = np.max(probabilities)
                        
                        # Only add points with significant probability (>0.3)
                        if max_prob > 0.3:
                            predicted_class = np.argmax(probabilities)
                            predicted_crime = label_encoder.inverse_transform([predicted_class])[0]
                            
                            # Create popup with prediction details
                            popup_text = f"""
                            <b>Time:</b> {time_name}<br>
                            <b>Predicted Crime:</b> {predicted_crime}<br>
                            <b>Probability:</b> {max_prob:.2%}<br>
                            <b>Borough:</b> {nearest_borough}
                            """
                            
                            # Add circle marker for the prediction
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=5,  # Small dot size
                                popup=folium.Popup(popup_text, max_width=300),
                                color=color,
                                fill=True,
                                fill_opacity=0.7
                            ).add_to(time_group)
                            
                            test_points.append([lat, lon, max_prob])
                            
                            # Print debug info for Manhattan
                            if nearest_borough == 'MANHATTAN':
                                print(f"Added Manhattan point at ({lat}, {lon}) with crime {predicted_crime}")
                    except Exception as e:
                        print(f"Error processing point ({lat}, {lon}): {str(e)}")
                        continue
            
            print(f"Added {len(test_points)} prediction points for {time_name}")
            # Add time group to map
            time_group.add_to(nyc_map)
        
        print("Adding layer control...")
        # Add layer control
        folium.LayerControl().add_to(nyc_map)
        
        # Save map with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'crime_maps/predictive_crime_map_{timestamp}.html'
        
        # Create directory if it doesn't exist
        os.makedirs('crime_maps', exist_ok=True)
        
        print(f"Saving map to {filename}...")
        try:
            # Save the map
            nyc_map.save(filename)
            
            # Verify the file was created
            if os.path.exists(filename):
                print(f"\nPredictive crime map saved successfully as: {filename}")
                print("File size:", os.path.getsize(filename), "bytes")
            else:
                print("Warning: File was not created")
                print("Current working directory:", os.getcwd())
                print("Attempted save path:", os.path.abspath(filename))
                
                # Try saving to current directory as a test
                test_filename = 'test_map.html'
                nyc_map.save(test_filename)
                if os.path.exists(test_filename):
                    print(f"Successfully saved test map to: {test_filename}")
                    os.remove(test_filename)
                else:
                    print("Failed to save test map")
        except Exception as e:
            print(f"Error saving map: {str(e)}")
            print("Current working directory:", os.getcwd())
            print("Attempted save path:", os.path.abspath(filename))
        
    except Exception as e:
        print(f"Error creating predictive crime map: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())

def execute_predictor(df, time_model, label_encoder, location=None, hour=None):
    """Execute the crime predictor for a specific location and time"""
    try:
        print("\nExecuting crime predictor...")
        
        if time_model is None or label_encoder is None:
            print("Error: Model or label encoder is not available")
            return
        
        # Default to Manhattan if no location provided
        if location is None:
            location = {
                'latitude': 40.7831,
                'longitude': -73.9712,
                'borough': 'MANHATTAN'
            }
        
        # Default to current hour if no hour provided
        if hour is None:
            import datetime
            hour = datetime.datetime.now().hour
        
        print(f"\nMaking prediction for:")
        print(f"Location: {location['borough']} ({location['latitude']}, {location['longitude']})")
        print(f"Hour: {hour}")
        
        # Create features for prediction
        features = pd.DataFrame({
            'hour_of_day': [hour],
            'is_night': [hour >= 20 or hour <= 5],
            'is_weekend': [False],  # Default to weekday
            'boro_nm': [location['borough']]
        })
        
        # Get prediction
        features_dummies = pd.get_dummies(features)
        # Ensure all required columns are present
        missing_cols = set(time_model.feature_names_in_) - set(features_dummies.columns)
        for col in missing_cols:
            features_dummies[col] = 0
        features_dummies = features_dummies[time_model.feature_names_in_]
        
        # Get predicted crime type
        predicted_class = time_model.predict(features_dummies)[0]
        predicted_crime = label_encoder.inverse_transform([predicted_class])[0]
        
        # Get prediction probabilities
        probabilities = time_model.predict_proba(features_dummies)[0]
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        top_5_crimes = label_encoder.inverse_transform(top_5_indices)
        top_5_probs = probabilities[top_5_indices]
        
        print("\nPrediction Results:")
        print(f"Most likely crime: {predicted_crime}")
        print("\nTop 5 predicted crimes:")
        for crime, prob in zip(top_5_crimes, top_5_probs):
            print(f"{crime}: {prob:.2%} probability")
        
        # Get historical data for comparison
        historical_data = df[
            (df['boro_nm'] == location['borough']) & 
            (df['hour_of_day'] == hour)
        ]
        
        if not historical_data.empty:
            print("\nHistorical Data for Comparison:")
            print(f"Number of crimes at this hour: {len(historical_data)}")
            print("\nMost common crimes at this hour:")
            crime_counts = historical_data['ofns_desc'].value_counts().head(5)
            for crime, count in crime_counts.items():
                print(f"{crime}: {count} incidents")
        
        return predicted_crime, top_5_crimes, top_5_probs
        
    except Exception as e:
        print(f"Error executing predictor: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return None, None, None

def generate_borough_predictions(df, time_model, label_encoder):
    """Generate detailed crime predictions for each borough and time period"""
    try:
        print("\nGenerating borough predictions...")
        
        # Get unique boroughs
        boroughs = df['boro_nm'].unique()
        print(f"Found boroughs: {', '.join(boroughs)}")
        
        # Define time periods
        hours = [0, 6, 12, 18]  # Midnight, 6 AM, Noon, 6 PM
        time_names = ['Midnight', '6 AM', 'Noon', '6 PM']
        
        # Create predictions for each borough and time
        predictions = {}
        
        for borough in boroughs:
            print(f"\nProcessing predictions for {borough}...")
            borough_predictions = {}
            
            for hour, time_name in zip(hours, time_names):
                # Create features for prediction
                features = pd.DataFrame({
                    'hour_of_day': [hour],
                    'is_night': [hour >= 20 or hour <= 5],
                    'is_weekend': [False],  # Default to weekday
                    'boro_nm': [borough]
                })
                
                # Get prediction
                features_dummies = pd.get_dummies(features)
                # Ensure all required columns are present
                missing_cols = set(time_model.feature_names_in_) - set(features_dummies.columns)
                for col in missing_cols:
                    features_dummies[col] = 0
                features_dummies = features_dummies[time_model.feature_names_in_]
                
                # Get prediction probabilities
                probabilities = time_model.predict_proba(features_dummies)[0]
                
                # Get top 5 predicted crimes
                top_5_indices = np.argsort(probabilities)[-5:][::-1]
                top_5_crimes = label_encoder.inverse_transform(top_5_indices)
                top_5_probs = probabilities[top_5_indices]
                
                # Store predictions
                borough_predictions[time_name] = {
                    'crimes': top_5_crimes,
                    'probabilities': top_5_probs
                }
            
            predictions[borough] = borough_predictions
        
        # Print detailed predictions in a clear format
        print("\nCRIME PREDICTIONS BY BOROUGH AND TIME")
        print("=" * 80)
        
        for borough, borough_data in predictions.items():
            print(f"\n{borough.upper()}")
            print("-" * 40)
            
            for time_name, time_data in borough_data.items():
                print(f"\n{time_name}:")
                for i, (crime, prob) in enumerate(zip(time_data['crimes'], time_data['probabilities']), 1):
                    print(f"  {i}. {crime}: {prob:.2%} probability")
        
        # Save predictions to a file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'predictions/crime_predictions_{timestamp}.txt'
        os.makedirs('predictions', exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("CRIME PREDICTIONS BY BOROUGH AND TIME\n")
            f.write("=" * 80 + "\n\n")
            
            for borough, borough_data in predictions.items():
                f.write(f"{borough.upper()}\n")
                f.write("-" * 40 + "\n\n")
                
                for time_name, time_data in borough_data.items():
                    f.write(f"{time_name}:\n")
                    for i, (crime, prob) in enumerate(zip(time_data['crimes'], time_data['probabilities']), 1):
                        f.write(f"  {i}. {crime}: {prob:.2%} probability\n")
                    f.write("\n")
        
        print(f"\nDetailed predictions saved to: {filename}")
        return predictions
        
    except Exception as e:
        print(f"Error generating borough predictions: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return None

def main():
    """Main execution function"""
    try:
        print("Fetching NYC crime data...")
        df = fetch_nyc_crime_data()
        
        print("Preprocessing data...")
        df = preprocess_data(df)
        
        print("Creating current crime heat map...")
        create_crime_heatmap(df)
        
        print("Training Random Forest model...")
        model = train_model(df)
        
        print("Analyzing crime time patterns...")
        time_model, label_encoder = predict_crime_times(df)
        
        if time_model is not None:
            print("Generating borough predictions...")
            predictions = generate_borough_predictions(df, time_model, label_encoder)
            
            print("Creating predictive crime map...")
            create_predictive_heatmap(df, time_model, label_encoder)
            
            # Execute predictor for different locations and times
            print("\nMaking sample predictions...")
            
            # Example locations
            locations = [
                {
                    'latitude': 40.7831,
                    'longitude': -73.9712,
                    'borough': 'MANHATTAN'
                },
                {
                    'latitude': 40.6782,
                    'longitude': -73.9442,
                    'borough': 'BROOKLYN'
                },
                {
                    'latitude': 40.7282,
                    'longitude': -73.7949,
                    'borough': 'QUEENS'
                }
            ]
            
            # Example times
            hours = [0, 6, 12, 18]  # Midnight, 6 AM, Noon, 6 PM
            
            for location in locations:
                print(f"\nPredictions for {location['borough']}:")
                for hour in hours:
                    execute_predictor(df, time_model, label_encoder, location, hour)
        
        # Save models
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(model, f'models/nyc_crime_predictor_{timestamp}.pkl')
        if time_model is not None:
            joblib.dump(time_model, f'models/crime_time_predictor_{timestamp}.pkl')
            joblib.dump(label_encoder, f'models/crime_label_encoder_{timestamp}.pkl')
        
        print("\nAll files saved successfully:")
        print(f"- Current crime map: crime_maps/current_crime_map_*.html")
        print(f"- Predictive crime map: crime_maps/predictive_crime_map_*.html")
        print(f"- Crime prediction model: models/nyc_crime_predictor_*.pkl")
        print(f"- Time prediction model: models/crime_time_predictor_*.pkl")
        print(f"- Label encoder: models/crime_label_encoder_*.pkl")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
