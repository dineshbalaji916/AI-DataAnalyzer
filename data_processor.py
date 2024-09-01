import pandas as pd

class DataProcessor:
    def load_data(self, file_path):
        """Load data from a file."""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")

    def clean_data(self, data):
        """Clean data (basic placeholder implementation)."""
        # Example cleaning step: drop rows with missing values
        return data.dropna()

    def preprocess_data(self, data):
        """Preprocess data (basic placeholder implementation)."""
        # Example preprocessing step: convert categorical variables to numeric
        return pd.get_dummies(data)
