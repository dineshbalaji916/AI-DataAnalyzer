from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AnalysisEngine:
    def __init__(self, data):
        self.data = data
        self.save_path = os.path.join(os.getcwd(), 'analysis_images')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def correlation_analysis(self):
        """Perform correlation analysis and save correlation heatmap as PNG."""
        correlation_matrix = self.data.corr()
        self._plot_correlation(correlation_matrix)
        return correlation_matrix

    def clustering_analysis(self, n_clusters=3):
        """Perform clustering analysis using KMeans."""
        kmeans = KMeans(n_clusters=n_clusters)
        self.data['Cluster'] = kmeans.fit_predict(self.data.select_dtypes(include=['float64', 'int64']))
        self._plot_clustering()
        return self.data, kmeans

    def regression_analysis(self, target_column):
        """Perform regression analysis."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        model = LinearRegression()
        model.fit(X, y)
        self._plot_regression(X, y, model)
        return model.coef_, model.intercept_

    def classification_analysis(self, target_column):
        """Perform classification analysis using RandomForest."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        model = RandomForestClassifier()
        model.fit(X, y)
        self._plot_feature_importance(model)
        return model.feature_importances_

    def descriptive_statistics(self):
        """Calculate descriptive statistics."""
        return self.data.describe()

    def anomaly_detection(self):
        """Perform anomaly detection using Isolation Forest."""
        model = IsolationForest(contamination=0.1)
        self.data['Anomaly'] = model.fit_predict(self.data.select_dtypes(include=['float64', 'int64']))
        anomalies = self.data[self.data['Anomaly'] == -1]
        self._plot_anomalies(anomalies)
        return anomalies

    def _plot_correlation(self, correlation_matrix):
        """Plot and save the correlation matrix heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix Heatmap')
        plt.savefig(os.path.join(self.save_path, 'Correlation_Matrix.png'))
        plt.close()

    def _plot_clustering(self):
        """Plot clustering results."""
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], c=self.data['Cluster'], cmap='viridis')
        plt.title('Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(os.path.join(self.save_path, 'Clustering_Results.png'))
        plt.close()

    def _plot_regression(self, X, y, model):
        """Plot regression results."""
        plt.scatter(X.iloc[:, 0], y, color='blue')
        plt.plot(X.iloc[:, 0], model.predict(X), color='red')
        plt.title('Regression Results')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.savefig(os.path.join(self.save_path, 'Regression_Results.png'))
        plt.close()

    def _plot_feature_importance(self, model):
        """Plot feature importance."""
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), indices)
        plt.xlabel('Importance')
        plt.savefig(os.path.join(self.save_path, 'Feature_Importance.png'))
        plt.close()

    def _plot_anomalies(self, anomalies):
        """Plot anomalies."""
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], color='blue', label='Normal')
        plt.scatter(anomalies.iloc[:, 0], anomalies.iloc[:, 1], color='red', label='Anomaly')
        plt.title('Anomaly Detection')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'Anomaly_Detection.png'))
        plt.close()

    def generate_insights(self, analysis_type, result):
        """Generate insights based on the analysis type and result."""
        insights = ""
        if analysis_type == 'correlation':
            strong_correlations = result[(result.abs() > 0.7) & (result != 1.0)].stack()
            if not strong_correlations.empty:
                insights = f"Strong correlations found between the following pairs:\n{strong_correlations}"
            else:
                insights = "No strong correlations found between any pairs of variables."
        elif analysis_type == 'clustering':
            num_clusters = len(np.unique(result['Cluster']))
            cluster_sizes = result['Cluster'].value_counts()
            insights = f"Clustering analysis identified {num_clusters} clusters. Cluster sizes are: \n{cluster_sizes.to_string()}"
        elif analysis_type == 'regression':
            coef, intercept = result
            insights = f"Regression analysis suggests that the relationship between the predictors and the target variable is linear. Coefficients of the model are: {coef} with an intercept of {intercept}."
        elif analysis_type == 'classification':
            top_features = np.argsort(result)[-3:][::-1]
            insights = f"Classification analysis indicates that the top 3 features influencing the target variable are: {top_features}."
        elif analysis_type == 'descriptive_stats':
            mean_values = result.loc['mean']
            insights = f"Descriptive statistics indicate the average values of the numerical features are: \n{mean_values.to_string()}"
        elif analysis_type == 'anomaly_detection':
            num_anomalies = len(result)
            insights = f"Anomaly detection identified {num_anomalies} anomalies. These outliers could indicate unusual behavior or errors in the data."
        return insights
