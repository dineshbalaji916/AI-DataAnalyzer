import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def load_data(self, file_path):
        """Load data from various file types."""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, Excel, or JSON file.")

    def clean_data(self, data):
        """Clean the data by handling missing values and removing duplicates."""
        # Remove duplicates
        data = data.drop_duplicates()
        # Handle missing values (drop rows with missing target values)
        data = data.dropna()
        return data

    def preprocess_data(self, data):
        """Preprocess the data by filling missing values and scaling numeric features."""
        # Separate numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns

        # Fill missing values for numeric columns with column mean
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Standardize numeric features
        scaler = StandardScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        return data
