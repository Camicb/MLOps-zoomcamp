# Import required libraries
import pickle
import pandas as pd
import os
import argparse

# Open trained model
def get_model(model_path):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

# Function to read and preprocess data for predictions 
def read_data(input_file):
    df = pd.read_parquet(input_file)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

# Make predictions
def get_predictions(df, model_path):
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    dv, model = get_model(model_path)
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

# Get results
def process_results(df, month, year, y_pred):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    output_file = f'C:/Users/Camila/OneDrive/Escritorio/mlops zoomcamp/HW4/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predictions'] = y_pred
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)
    return df_result

def apply_model(month, year, model_path):
    input_file = f'C:/Users/Camila/OneDrive/Escritorio/mlops zoomcamp/data/yellow/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)
    y_pred = get_predictions(df, model_path)  
    return process_results(df, month, year, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--month", type=int, required=True, help="month with the format M")
    parser.add_argument("--year", type=int, required=True, help="year with the format yyyy")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    args = parser.parse_args()
    month = args.month
    year = args.year
    model_path = args.model_path
    model_path = os.path.join(model_path)
    results = apply_model(month, year, model_path)
    print(results['predictions'].mean())
    