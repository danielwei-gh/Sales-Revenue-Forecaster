import pickle
import argparse
import pandas as pd
import xgboost as xgb

# Parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--date',
                    required=True,
                    type=str,
                    help='The operational date for the sales revenue.')

parser.add_argument('--venue_id',
                    required=True,
                    type=str,
                    help='Unique identifier for the venue.')

parser.add_argument('--concept',
                    required=True,
                    type=str,
                    help='The type or category of the venue.')

parser.add_argument('--city',
                    required=True,
                    type=str,
                    help='The city where the venue is located.')

parser.add_argument('--country',
                    required=True,
                    type=str,
                    help='The country where the venue is located.')

args = parser.parse_args()

business_date = args.date
venue_xref_id = args.venue_id
concept = args.concept
city = args.city
country = args.country

input = pd.DataFrame(
    {'business_date': [pd.Timestamp(business_date)],
     'venue_xref_id': [venue_xref_id],
     'concept': [concept],
     'city': [city],
     'country': [country]}
)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for the data.
    """
    df = df.copy()

    # Add new features
    df['day_of_week'] = df['business_date'].dt.day_of_week
    df['day_of_year'] = df['business_date'].dt.day_of_year
    df['month'] = df['business_date'].dt.month
    df['quarter'] = df['business_date'].dt.quarter

    return df

def encode_categorical(df: pd.DataFrame, encoder, features: list) -> pd.DataFrame:
    df = df.copy()

    # Encode categorical data
    df = encoder.transform(df)

    # Specify categorical data type for XGBoost
    df[features] = df[features].astype('category')

    return df

# Load the encoder to transform the input data
with open('encoder.obj', 'rb') as f:
    encoder = pickle.load(f)


FEATURES = ['venue_xref_id', 'concept', 'city', 'country', 
            'day_of_week', 'day_of_year', 'month', 'quarter']

input = create_features(input)
input.drop(columns='business_date', inplace=True)

input = encode_categorical(input, encoder, FEATURES)

forecaster = xgb.XGBRegressor()
forecaster.load_model('model.json')

sales_revenue_with_tax = forecaster.predict(input)
print(f'Forecasted Sales Revenue for {venue_xref_id}:')
print(f'{business_date}: ${sales_revenue_with_tax.item():.2f}')