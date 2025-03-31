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

def fetch_nyc_crime_data():
    """Fetch crime data from NYC Open Data API"""
    try:
        url = "https://data.cityofnewyork.us/resource/5uac-w243.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            print("\nData fetched successfully")
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
        'law_cat_cd',      # Crime category
        'vic_age_group',   # Victim age group
        'vic_sex',         # Victim sex
        'ofns_desc'        # Offense description
    ]
    
    # Verify and handle missing columns
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        features = [col for col in features if col not in missing_cols]
    
    # Drop rows with missing values
    df = df.dropna(subset=features)
    
    # Convert latitude and longitude to float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    return df

def create_crime_map(df, model, label_encoders):
    """Create and save an interactive map with crime predictions"""
    try:
        # Create base map centered on NYC
        nyc_map = folium.Map(location=[40.7128, -74.0060], 
                            zoom_start=11,
                            tiles='CartoDB positron')
        
        # Create a feature group for each crime type
        crime_groups = {}
        
        # Create a color map for different crime types
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                'darkpurple', 'pink', 'lightblue', 'lightgreen']
        
        # Process each location
        for idx, row in df.iterrows():
            try:
                lat, lon = float(row['latitude']), float(row['longitude'])
                
                # Skip if coordinates are invalid
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                
                # Prepare features for prediction
                features = row[['addr_pct_cd', 'boro_nm', 'vic_age_group', 'vic_sex']].copy()
                for col in features.index:
                    if col in label_encoders:
                        features[col] = label_encoders[col].transform([str(features[col])])[0]
                
                # Make prediction
                prediction = model.predict([features])[0]
                crime_type = prediction
                
                # Create feature group for this crime type if it doesn't exist
                if crime_type not in crime_groups:
                    crime_groups[crime_type] = folium.FeatureGroup(name=crime_type)
                    
                # Add a circle marker for this location
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    popup=f"Predicted Crime: {crime_type}<br>" +
                          f"Borough: {row['boro_nm']}<br>" +
                          f"Precinct: {row['addr_pct_cd']}",
                    color=colors[len(crime_groups) % len(colors)],
                    fill=True,
                    fillOpacity=0.7
                ).add_to(crime_groups[crime_type])
                
            except Exception as e:
                print(f"Error processing location {idx}: {str(e)}")
                continue
        
        # Add all feature groups to the map
        for crime_type, group in crime_groups.items():
            group.add_to(nyc_map)
        
        # Add layer control to toggle crime types
        folium.LayerControl().add_to(nyc_map)
        
        # Save map to HTML file
        nyc_map.save('nyc_crime_predictions.html')
        print("\nCrime prediction map saved as 'nyc_crime_predictions.html'")
        
        # Print prediction summary
        print("\nPredicted Crime Types Summary:")
        for crime_type, group in crime_groups.items():
            print(f"{crime_type}: {len(group._children)} predicted locations")
            
    except Exception as e:
        print(f"Error creating map: {str(e)}")

def train_model(df):
    """Train and evaluate the crime prediction model"""
    # Prepare features and target
    features = ['addr_pct_cd', 'boro_nm', 'vic_age_group', 'vic_sex']
    X = df[features]
    y = df['law_cat_cd']
    
    # Initialize dictionary to store label encoders
    label_encoders = {}
    
    # Encode categorical variables
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, label_encoders

def main():
    """Main execution function"""
    try:
        print("Fetching NYC crime data...")
        df = fetch_nyc_crime_data()
        
        print("Preprocessing data...")
        df = preprocess_data(df)
        
        print("Training prediction model...")
        model, label_encoders = train_model(df)
        
        print("Creating crime prediction map...")
        create_crime_map(df, model, label_encoders)
        
        # Save model and encoders
        joblib.dump(model, 'nyc_crime_predictor.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        print("\nModel saved as 'nyc_crime_predictor.pkl'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()