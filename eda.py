#EDA script for data analysis of the Lead Scoring csv file in data folder
import numpy as np
import pandas as pd
def load_data(file_path):
    """
    Load the dataset from the specified file path.
    
    Parameters:
    file_path (str): The path to the CSV file containing the dataset.
    
    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def analyze_data(data):
    """
    Perform basic analysis on the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset to analyze.
    
    Returns:
    None
    """
    if data is not None:
        print("Data Overview:")
        print(data.head())
        print("\nData Description:")
        print(data.describe())
        print("\nMissing Values:")
        print(data.isnull().sum())
        print("\nData Types:")
        print(data.dtypes)
    else:
        print("No data to analyze.")

def main():
    file_path = 'D:/internpro/AI-Powered-Lead-Scoring-System-for-CRMs/data/Lead Scoring.csv'  # Adjust the path as necessary
    data = load_data(file_path)
    analyze_data(data)

if __name__ == "__main__":
    main()