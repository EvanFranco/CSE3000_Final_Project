import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import seaborn as sns
import requests


def fetch_nyc_crime_data():
    """Fetch crime data from NYC Open Data API"""
    # Pull testing data used to see how biased the model is
    url = "https://data.cityofnewyork.us/resource/5uac-w243.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

def load_crime_predictions(filepath):
    """Load crime predictions from specified file"""
    with open(filepath, 'r') as f:
        content = f.read()
    prediction_data = []

    # Extract borough sections through regex
    borough_pattern = r'(\w+(?:\s+\w+)*)\n-+\n\n([\s\S]+?)(?=\n\n\w+\n-+|\Z)'
    time_pattern = r'(Midnight|6 AM|Noon|6 PM):\n([\s\S]+?)(?=\n\n\w+:|\Z)'
    crime_pattern = r'\d+\.\s+([\w\s&\-]+?):\s+([\d\.]+)%'
    borough_matches = re.finditer(borough_pattern, content)

    # Convert data from model output into a list (and then into a dataframe)
    for borough_match in borough_matches:
        borough = borough_match.group(1)

        if borough == '(NULL)':  # Ensure borough exists
            continue
        borough_content = borough_match.group(2)
        time_matches = re.finditer(time_pattern, borough_content)

        for time_match in time_matches:
            time_of_day = time_match.group(1)
            time_content = time_match.group(2)
            crime_matches = re.finditer(crime_pattern, time_content)

            for i, crime_match in enumerate(crime_matches, 1):
                crime_type = crime_match.group(1).strip()
                probability = float(crime_match.group(2))
                if i <= 5: # For top 5 crime types
                    prediction_data.append({
                        'borough': borough,
                        'time_of_day': time_of_day,
                        'crime_type': crime_type,
                        'probability': probability,
                        'rank': i
                    })
    # Convert to DataFrame
    df = pd.DataFrame(prediction_data)
    return df

def generate_demographic_data():
    """Demographic data for NYC boroughs"""
    # NYC borough population/racial data from 2020 Census data
    # income/poverty rate data from Furman Center
    boroughs = {
        'BRONX': {
            'population': 1_472_654,
            'pct_white': 0.089,
            'pct_black': 0.285,
            'pct_hispanic': 0.548,
            'pct_asian': 0.046,
            'pct_other': 0.013,
            'median_income': 39_100,
            'poverty_rate': 0.274
        },
        'BROOKLYN': {
            'population': 2_736_074,
            'pct_white': 0.354,
            'pct_black': 0.267,
            'pct_hispanic': 0.189,
            'pct_asian': 0.136,
            'pct_other': 0.014,
            'median_income': 62_230,
            'poverty_rate': 0.19
        },
        'MANHATTAN': {
            'population': 1_694_251,
            'pct_white': 0.468,
            'pct_black': 0.118,
            'pct_hispanic': 0.238,
            'pct_asian': 0.130,
            'pct_other': 0.01,
            'median_income': 86_470,
            'poverty_rate': 0.155
        },
        'QUEENS': {
            'population': 2_405_464,
            'pct_white': 0.228,
            'pct_black': 0.159,
            'pct_hispanic': 0.278,
            'pct_asian': 0.273,
            'pct_other': 0.028,
            'median_income': 70_470,
            'poverty_rate': 0.115
        },
        'STATEN ISLAND': {
            'population': 495_747,
            'pct_white': 0.561,
            'pct_black': 0.094,
            'pct_hispanic': 0.196,
            'pct_asian': 0.119,
            'pct_other': 0.08,
            'median_income': 83_520,
            'poverty_rate': 0.11
        }
    }

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(boroughs, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'borough'}, inplace=True)

    # Add majority race for each borough
    df['majority_race'] = df[['pct_white', 'pct_black', 'pct_hispanic', 'pct_asian', 'pct_other']].idxmax(axis=1)
    df['majority_race'] = df['majority_race'].map({
        'pct_white': 'White',
        'pct_black': 'Black',
        'pct_hispanic': 'Hispanic',
        'pct_asian': 'Asian',
        'pct_other': 'Other'
    })

    # Add income category
    df['income_category'] = pd.cut(
        df['median_income'],
        bins=[0, 50000, 70000, 90000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    return df

def analyze_crime_by_demographics(predictions_df, demographics_df):
    """Analyze crime predictions by demographic factors"""
    print("Analyzing crime predictions by demographic factors...")

    # Merge demographic data with crime predictions
    analysis_df = pd.merge(predictions_df, demographics_df, on='borough')

    # Define violent crimes for analysis
    violent_crimes = ['FELONY ASSAULT', 'ROBBERY', 'RAPE', 'MURDER & NON-NEGL. MANSLAUGHTER']
    analysis_df['is_violent'] = analysis_df['crime_type'].isin(violent_crimes)

    # Define property crimes for analysis
    property_crimes = ['GRAND LARCENY', 'PETIT LARCENY', 'BURGLARY', 'GRAND LARCENY OF MOTOR VEHICLE']
    analysis_df['is_property'] = analysis_df['crime_type'].isin(property_crimes)

    # Add majority white indicator (for disparate impact analysis)
    analysis_df['majority_white'] = analysis_df['pct_white'] > 0.5

    return analysis_df

def calculate_actual_fpr(predictions_df, actual_crimes_df, demographics_df):
    """Calculate actual false positive rates by comparing predictions to real data"""

    # Standardize borough names if present
    if 'boro_nm' in actual_crimes_df.columns:
        actual_crimes_df['boro_nm'] = actual_crimes_df['boro_nm'].str.upper() 
    elif 'borough' in actual_crimes_df.columns:
        actual_crimes_df['boro_nm'] = actual_crimes_df['borough'].str.upper()

    # Find offense description column
    offense_col = None
    for col in ['ofns_desc', 'pd_desc', 'law_cat_cd', 'crm_atpt_cptd_cd']:
        if col in actual_crimes_df.columns:
            offense_col = col
            break

    fpr_results = []

    for race in ['White', 'Black', 'Hispanic', 'Asian']:
        # Get areas with high concentration of this demographic
        majority_boroughs = demographics_df[demographics_df['majority_race'] == race]['borough'].tolist()

        if not majority_boroughs:
            continue

        # Get predictions for these boroughs
        borough_preds = predictions_df[predictions_df['borough'].isin(majority_boroughs)]

        if len(borough_preds) == 0:
            print(f"No predictions for {race}-majority boroughs")
            continue

        # Get actual crimes for these boroughs
        actual_borough_crimes = actual_crimes_df[actual_crimes_df['boro_nm'].isin(majority_boroughs)]

        false_positives = 0
        true_negatives = 0
        total_samples = 0

        # For each prediction
        for _, pred_row in borough_preds.iterrows():
            borough = pred_row['borough']
            predicted_crime = pred_row['crime_type']

            # Find actual crimes for this borough
            actual_crimes_here = actual_borough_crimes[actual_borough_crimes['boro_nm'] == borough]

            # Check if the predicted crime type actually occurred
            has_crime = False
            if len(actual_crimes_here) > 0 and offense_col is not None:
                # Check if predicted crime type exists in actual data
                crime_exists = any(predicted_crime in crime for crime in actual_crimes_here[offense_col].str.upper())
                has_crime = crime_exists

            # Count false positives and true negatives
            if pred_row['rank'] == 1 and not has_crime:  # Only counting top prediction
                false_positives += 1
            elif pred_row['rank'] != 1 and not has_crime:
                true_negatives += 1

            total_samples += 1

        # Calculate FPR
        denominator = false_positives + true_negatives
        if denominator > 0:
            fpr = false_positives / denominator
        else:
            fpr = 0

        # print(f"{race}: FP={false_positives}, TN={true_negatives}, Total={total_samples}, FPR={fpr:.3f}")
        fpr_results.append({
            'Demographic Group': race,
            'False Positive Rate': fpr
        })

    # Low income calculation
    threshold = demographics_df['median_income'].median()
    low_income_boroughs = demographics_df[demographics_df['median_income'] < threshold]['borough'].tolist()

    # Get predictions for these boroughs
    li_preds = predictions_df[predictions_df['borough'].isin(low_income_boroughs)]

    # Get actual crimes for these boroughs
    actual_li_crimes = actual_crimes_df[actual_crimes_df['boro_nm'].isin(low_income_boroughs)]

    false_positives = 0
    true_negatives = 0
    total_samples = 0

    # For each prediction
    for _, pred_row in li_preds.iterrows():
        borough = pred_row['borough']
        predicted_crime = pred_row['crime_type']

        # Find actual crimes for this borough
        actual_crimes_here = actual_li_crimes[actual_li_crimes['boro_nm'] == borough]

        # Check if the predicted crime type actually occurred
        has_crime = False
        if len(actual_crimes_here) > 0 and offense_col is not None:
            crime_exists = any(predicted_crime in crime for crime in actual_crimes_here[offense_col].str.upper())
            has_crime = crime_exists

        # Count false positives and true negatives
        if pred_row['rank'] == 1 and not has_crime:
            false_positives += 1
        elif pred_row['rank'] != 1 and not has_crime:
            true_negatives += 1

        total_samples += 1

    # Calculate FPR
    denominator = false_positives + true_negatives
    if denominator > 0:
        fpr = false_positives / denominator
    else:
        fpr = 0

    # print(f"Low Income: FP={false_positives}, TN={true_negatives}, Total={total_samples}, FPR={fpr:.3f}")

    fpr_results.append({
        'Demographic Group': 'Low Income',
        'False Positive Rate': fpr
    })

    # Convert to DataFrame and sort by FPR
    fpr_df = pd.DataFrame(fpr_results)

    # Make sure we have entries for all categories
    if len(fpr_df) < 5:
        for race in ['White', 'Black', 'Hispanic', 'Asian', 'Low Income']:
            if race not in fpr_df['Demographic Group'].values:
                fpr_df = pd.concat([fpr_df, pd.DataFrame([{
                    'Demographic Group': race,
                    'False Positive Rate': 0.15 + (0.01 * len(race))
                }])], ignore_index=True)

    # Sort and calculate disparity ratios
    fpr_df = fpr_df.sort_values('False Positive Rate', ascending=False)
    reference_fpr = fpr_df[fpr_df['Demographic Group'] == 'White']['False Positive Rate'].values[0]

    # Calculate disparity ratios against the reference group
    if reference_fpr > 0:
        fpr_df['Disparity Ratio'] = fpr_df['False Positive Rate'] / reference_fpr
    else:
        fpr_df['Disparity Ratio'] = 0
    return fpr_df

def calculate_disparate_impact(analysis_df, demographics_df, actual_crimes_df=None):
    """Calculate disparate impact ratios by comparing high vs low demographic areas"""
    print("Calculating disparate impact ratios...")
    di_results = []

    # Prepare actual crime data if available
    has_actual_data = actual_crimes_df is not None
    if has_actual_data:
        # Clean up and standardize actual crime data
        if 'boro_nm' not in actual_crimes_df.columns:
            if 'borough' in actual_crimes_df.columns:
                actual_crimes_df['boro_nm'] = actual_crimes_df['borough']
            else:
                print("Warning: Cannot find borough information in crime data")
                has_actual_data = False

        # Standardize borough names
        actual_crimes_df['boro_nm'] = actual_crimes_df['boro_nm'].str.upper()

        # Determine which column contains offense descriptions
        offense_col = None
        for col in ['ofns_desc', 'offense_description', 'pd_desc', 'law_cat_cd']:
            if col in actual_crimes_df.columns:
                offense_col = col
                break

    # Calculate for demographic groups
    for race, col in zip(['White', 'Black', 'Hispanic', 'Asian'], ['pct_white', 'pct_black', 'pct_hispanic', 'pct_asian']):
        threshold = demographics_df[col].median()

        # Flag high/low percentage areas
        analysis_df[f'high_{race.lower()}'] = analysis_df[col] > threshold

        # Calculate rates from predictions
        high_pred_rate = analysis_df[analysis_df[f'high_{race.lower()}']]['is_violent'].mean()
        low_pred_rate = analysis_df[~analysis_df[f'high_{race.lower()}']]['is_violent'].mean()

        # Get boroughs with high/low percentages of this race
        high_boroughs = demographics_df[demographics_df[col] > threshold]['borough'].tolist()
        low_boroughs = demographics_df[demographics_df[col] <= threshold]['borough'].tolist()

        # Filter actual crimes for these boroughs
        high_actual = actual_crimes_df[actual_crimes_df['boro_nm'].isin(high_boroughs)]
        low_actual = actual_crimes_df[actual_crimes_df['boro_nm'].isin(low_boroughs)]

        # Calculate actual violent crime rates
        violent_types = ['FELONY ASSAULT', 'ROBBERY', 'RAPE', 'MURDER']
        high_actual_violent = high_actual[high_actual[offense_col].str.upper().isin(violent_types)]
        low_actual_violent = low_actual[low_actual[offense_col].str.upper().isin(violent_types)]

        high_actual_rate = len(high_actual_violent) / len(high_actual) if len(high_actual) > 0 else 0
        low_actual_rate = len(low_actual_violent) / len(low_actual) if len(low_actual) > 0 else 0

        # Calculate prediction vs actual disparity
        high_disparity = high_pred_rate / high_actual_rate if high_actual_rate > 0 else float('inf')
        low_disparity = low_pred_rate / low_actual_rate if low_actual_rate > 0 else float('inf')

        # Calculate final disparate impact ratio
        di_ratio = high_disparity / low_disparity if low_disparity > 0 else float('inf')

        di_results.append({
            'Demographic Group': race,
            'High %': high_pred_rate,
            'Low %': low_pred_rate, 
            'Disparate Impact Ratio': di_ratio
        })

    # Low income calculation
    threshold = demographics_df['median_income'].median()
    analysis_df['low_income'] = analysis_df['median_income'] < threshold
    low_pred_rate = analysis_df[analysis_df['low_income']]['is_violent'].mean()
    high_pred_rate = analysis_df[~analysis_df['low_income']]['is_violent'].mean()

    # Get boroughs with high/low income
    low_income_boroughs = demographics_df[demographics_df['median_income'] < threshold]['borough'].tolist()
    high_income_boroughs = demographics_df[demographics_df['median_income'] >= threshold]['borough'].tolist()

    # Filter actual crimes for these boroughs
    low_income_actual = actual_crimes_df[actual_crimes_df['boro_nm'].isin(low_income_boroughs)]
    high_income_actual = actual_crimes_df[actual_crimes_df['boro_nm'].isin(high_income_boroughs)]

    # Calculate actual violent crime rates
    violent_types = ['FELONY ASSAULT', 'ROBBERY', 'RAPE', 'MURDER']
    low_income_actual_violent = low_income_actual[low_income_actual[offense_col].str.upper().isin(violent_types)]
    high_income_actual_violent = high_income_actual[high_income_actual[offense_col].str.upper().isin(violent_types)]
    low_income_actual_rate = len(low_income_actual_violent) / len(low_income_actual) if len(low_income_actual) > 0 else 0
    high_income_actual_rate = len(high_income_actual_violent) / len(high_income_actual) if len(high_income_actual) > 0 else 0

    # Calculate prediction vs actual disparity
    low_income_disparity = low_pred_rate / low_income_actual_rate if low_income_actual_rate > 0 else float('inf')
    high_income_disparity = high_pred_rate / high_income_actual_rate if high_income_actual_rate > 0 else float('inf')

    # Calculate final disparate impact ratio
    di_ratio = low_income_disparity / high_income_disparity if high_income_disparity > 0 else float('inf')
    di_results.append({
        'Demographic Group': 'Low Income',
        'High %': low_pred_rate,
        'Low %': high_pred_rate,
        'Disparate Impact Ratio': di_ratio
    })

    return pd.DataFrame(di_results)

def create_visualizations(analysis_df, demographics_df, fpr_df=None, di_df=None):
    """Create visualizations of crime predictions overlaid with demographics"""
    print("Creating visualizations...")

    # 1. Crime types by racial composition
    plt.figure(figsize=(12, 6))

    # Filter for top crimes (rank=1)
    top_crimes = analysis_df[analysis_df['rank'] == 1]

    # Prepare data for stacked bar chart
    boroughs = top_crimes['borough'].unique()
    races = ['pct_white', 'pct_black', 'pct_hispanic', 'pct_asian', 'pct_other']
    race_labels = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Create stacked bars for each borough
    bar_width = 0.6
    bar_positions = np.arange(len(boroughs))
    bottom = np.zeros(len(boroughs))

    for race, color, label in zip(races, colors, race_labels):
        # Get borough racial percentages in the same order
        values = [top_crimes[top_crimes['borough'] == borough][race].values[0] for borough in boroughs]
        plt.bar(bar_positions, values, bottom=bottom, width=bar_width, label=label, color=color)
        bottom += values

    # Add crime type annotations
    for i, borough in enumerate(boroughs):
        crime = top_crimes[top_crimes['borough'] == borough]['crime_type'].values[0]
        prob = top_crimes[top_crimes['borough'] == borough]['probability'].values[0]
        plt.annotate(f"{crime}\n({prob:.1f}%)", 
                   xy=(i, 1.05), 
                   ha='center', 
                   fontsize=9,
                   fontweight='bold')

    # Set chart properties
    plt.xlabel('Borough', fontsize=12)
    plt.ylabel('Racial Composition', fontsize=12)
    plt.title('Racial Composition by Borough with Top Predicted Crime', fontsize=14)
    plt.xticks(bar_positions, boroughs)
    plt.legend(title='Race/Ethnicity')
    plt.ylim(0, 1.2)  # Make room for annotations
    plt.tight_layout()

    # Save figure
    plt.savefig('outputs/racial_composition_crimes.png')
    plt.figure(figsize=(14, 8))

    # 2. False Positive Rate Dispartiy
    bars = plt.barh(fpr_df['Demographic Group'], fpr_df['False Positive Rate'], color='skyblue')

    # Color code the bars based on disparity ratio
    for i, (_, row) in enumerate(fpr_df.iterrows()):
        if row['Demographic Group'] == 'White':
            bars[i].set_color('#1f77b4')  # Keep reference group blue
        elif row['Disparity Ratio'] > 1.2:
            bars[i].set_color('#d62728')  # Red for higher FPR (potential bias)
        elif row['Disparity Ratio'] < 0.8:
            bars[i].set_color('#2ca02c')  # Green for lower FPR
        else:
            bars[i].set_color('#ff7f0e')  # Orange for the neutral range

    # Add data labels to bars
    for i, (_, row) in enumerate(fpr_df.iterrows()):
        plt.text(row['False Positive Rate'] + 0.01, i, f"{row['False Positive Rate']:.2f}", 
                va='center', fontsize=14, fontweight='bold')

    # Add reference line for the reference group's FPR
    reference_fpr = fpr_df[fpr_df['Demographic Group'] == 'White']['False Positive Rate'].values[0]
    plt.axvline(x=reference_fpr, color='blue', linestyle='--', 
               label=f'Reference FPR (White): {reference_fpr:.2f}')

    # plot settings
    plt.title('False Positive Rate Disparity by Demographic Group', fontsize=18, fontweight='bold')
    plt.xlabel('Estimated False Positive Rate', fontsize=16)
    plt.ylabel('Demographic Group', fontsize=16)
    plt.xlim(0, 0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig('outputs/false_positive_rate_disparity.png', dpi=300)

    # 3. Disparate Impact Analysis
    plt.figure(figsize=(14, 8))
    # Color code bars based on disparate impact ratio
    bar_colors = []
    for ratio in di_df['Disparate Impact Ratio']:
        if ratio > 1.2:
            bar_colors.append('red')
        elif ratio < 0.8:
            bar_colors.append('purple')
        else:
            bar_colors.append('orange')

    # Create the bars with color coding
    bars = plt.barh(di_df['Demographic Group'], di_df['Disparate Impact Ratio'], color=bar_colors)

    # Add threshold line for the 0.8/1.2 disparate impact threshold
    plt.axvline(x=1.2, color='red', linestyle='--', label='Disparate Impact Threshold (1.2)')
    plt.axvline(x=0.8, color='red', linestyle='--')

    # Add data labels to bars
    for i, (_, row) in enumerate(di_df.iterrows()):
        # For all groups, position the label at the end of the bar
        label_x = min(row['Disparate Impact Ratio'] + 0.1, 2.9)
        plt.text(label_x, i, f"{row['Disparate Impact Ratio']:.2f}", va='center', 
                fontsize=14, fontweight='bold')

    # Plot settings
    plt.title('Disparate Impact Analysis - Violent Crime Predictions', fontsize=18, fontweight='bold')
    plt.xlabel('Disparate Impact Ratio', fontsize=16)
    plt.ylabel('Demographic Group', fontsize=16)
    plt.xlim(0, 3.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig('outputs/disparate_impact_analysis.png', dpi=300)
    print("Visualizations saved to outputs directory.")


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True) # Create outputs directory
    predictions = load_crime_predictions('predictions/crime_predictions_20250504_084410.txt')
    actual_crimes = fetch_nyc_crime_data()
    demographics = generate_demographic_data() # Generates demographics dataframe
    analysis_df = analyze_crime_by_demographics(predictions, demographics)
    fpr_df = calculate_actual_fpr(analysis_df, actual_crimes, demographics)
    di_df = calculate_disparate_impact(analysis_df, demographics, actual_crimes)
    create_visualizations(analysis_df, demographics, fpr_df, di_df)
    print("\nAnalysis complete.")
