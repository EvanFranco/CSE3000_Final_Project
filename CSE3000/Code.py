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
        'occur_time',      # Time of occurrence
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
    
    # Handle occur_time if it exists
    if 'occur_time' in df.columns:
        try:
            # First, print a sample of occur_time values for debugging
            print("Sample occur_time values:", df['occur_time'].head())
            
            # Try to convert occur_time to hour, handling different possible formats
            df['occur_time'] = pd.to_datetime(df['occur_time'], format='%H:%M', errors='coerce')
            if df['occur_time'].isna().all():
                # If first attempt failed, try different format
                df['occur_time'] = pd.to_datetime(df['occur_time'], errors='coerce')
            df['occur_time'] = df['occur_time'].dt.hour
        except Exception as e:
            print(f"Error processing occur_time: {str(e)}")
            # If processing fails, create a dummy hour value
            df['occur_time'] = 0
    
    return df[features]

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
        
        # Save map to HTML file
        nyc_map.save('nyc_crime_map.html')
        print("\nCrime map saved as 'nyc_crime_map.html'")
        
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

def main():
    """Main execution function"""
    try:
        print("Fetching NYC crime data...")
        df = fetch_nyc_crime_data()
        
        print("Preprocessing data...")
        df = preprocess_data(df)
        
        print("Creating heat map...")
        create_crime_heatmap(df)
        
        print("Training Random Forest model...")
        model = train_model(df)
        
        # Save model
        joblib.dump(model, 'nyc_crime_predictor.pkl')
        print("\nModel saved as 'nyc_crime_predictor.pkl'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
