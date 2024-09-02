import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error


class AnalysisEngine:
    def __init__(self, data):
        self.data = data

    def correlation_analysis(self):
        """Perform correlation analysis and visualize the correlation matrix."""
        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=['number'])
    
        if numeric_data.empty:
            print("No numeric columns available for correlation analysis.")
            return
    
        correlation_matrix = numeric_data.corr(method='spearman')

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.show()

        insights = correlation_matrix.unstack().sort_values(ascending=False)
        print("Insights:")
        print(insights)
        return insights


    def clustering_analysis(self, algorithm='kmeans', n_clusters=2):
        """Perform clustering analysis using the specified algorithm."""
        # Drop non-numeric columns
        data_numeric = self.data.select_dtypes(include=['float64', 'int64'])
        
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters)
        elif algorithm == 'dbscan':
            model = DBSCAN()
        else:
            raise ValueError("Unsupported algorithm. Choose 'kmeans' or 'dbscan'.")

        clusters = model.fit_predict(data_numeric)
        self.data['Cluster'] = clusters
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=data_numeric.iloc[:, 0], y=data_numeric.iloc[:, 1], hue=self.data['Cluster'], palette='viridis')
        plt.title(f'{algorithm.capitalize()} Clustering')
        plt.show()

        # Insights
        print(f"Insights: Data has been clustered into {n_clusters} clusters using {algorithm}.")

    def regression_analysis(self, target_column, model_type='ridge'):
        """Perform regression analysis using the specified model."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Handle categorical features by encoding them
        X = pd.get_dummies(X)

        if model_type == 'ridge':
            model = Ridge()
        elif model_type == 'lasso':
            model = Lasso()
        else:
            raise ValueError("Unsupported model type. Choose 'ridge' or 'lasso'.")

        model.fit(X, y)
        predictions = model.predict(X)

        # Calculate Mean Squared Error
        mse = mean_squared_error(y, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=y, y=predictions)
        plt.title(f'{model_type.capitalize()} Regression')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.show()

        # Insights
        print(f"Insights: {model_type.capitalize()} regression model has been trained and evaluated.")
        
        return mse

    def classification_analysis(self, target_column):
        """Perform classification analysis."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Handle categorical features by encoding them
        X = pd.get_dummies(X)

        model = RandomForestClassifier()
        model.fit(X, y)
        predictions = model.predict(X)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=y, y=predictions)
        plt.title('Classification Results')
        plt.xlabel('True Labels')
        plt.ylabel('Predictions')
        plt.show()

        # Insights
        print("Insights: Classification model has been trained and evaluated.")

    def descriptive_statistics(self):
        """Calculate and display descriptive statistics."""
        stats_summary = self.data.describe()
        print("Descriptive Statistics:\n", stats_summary)

        # Insights
        print("Insights: Descriptive statistics calculated for the numerical features.")

    def anomaly_detection(self):
        """Perform anomaly detection using Z-score."""
        data_numeric = self.data.select_dtypes(include=['float64', 'int64'])
        z_scores = np.abs(stats.zscore(data_numeric))
        anomalies = (z_scores > 3).astype(int)

        self.data['Anomaly'] = anomalies.max(axis=1)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=self.data.index, y=self.data['Total'], hue=self.data['Anomaly'], palette='coolwarm')
        plt.title('Anomaly Detection')
        plt.show()

        # Insights
        print("Insights: Anomaly detection completed. Outliers have been marked.")
