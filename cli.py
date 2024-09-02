import os
from data_processor import DataProcessor
from analysis_engine import AnalysisEngine

def main():
    # Load and preprocess data
    file_path = input("Enter the file path (CSV, Excel, JSON): ")
    if not os.path.exists(file_path):
        print("File does not exist. Please check the path.")
        return

    processor = DataProcessor()
    data = processor.load_data(file_path)
    data = processor.clean_data(data)
    data = processor.preprocess_data(data)

    # Create an instance of AnalysisEngine
    analysis_engine = AnalysisEngine(data)

    while True:
        print("\nAI Employee Analysis Tool")
        print("=" * 40)
        print("1. Correlation Analysis")
        print("2. Clustering Analysis")
        print("3. Regression Analysis")
        print("4. Classification Analysis")
        print("5. Descriptive Statistics")
        print("6. Anomaly Detection")
        print("7. Exit")

        choice = input("Choose an analysis option (1-7): ")

        if choice == '1':
            print("Performing Correlation Analysis...")
            insights = analysis_engine.correlation_analysis()
            print("Insights:", insights)

        elif choice == '2':
            algo = input("Enter clustering algorithm (kmeans/dbscan): ")
            n_clusters = int(input("Enter number of clusters (for kmeans): ")) if algo == 'kmeans' else None
            print("Performing Clustering Analysis...")
            analysis_engine.clustering_analysis(n_clusters=n_clusters, algorithm=algo)

        elif choice == '3':
            target_column = input("Enter the target column for regression: ")
            model_type = input("Enter regression model type (ridge/lasso): ")
            print("Performing Regression Analysis...")
            mse = analysis_engine.regression_analysis(target_column, model_type=model_type)
            print(f"Mean Squared Error: {mse:.2f}")

        elif choice == '4':
            target_column = input("Enter the target column for classification: ")
            print("Performing Classification Analysis...")
            analysis_engine.classification_analysis(target_column)

        elif choice == '5':
            print("Calculating Descriptive Statistics...")
            stats = analysis_engine.descriptive_statistics()
            print("Descriptive Statistics:", stats)

        elif choice == '6':
            print("Performing Anomaly Detection...")
            analysis_engine.anomaly_detection()

        elif choice == '7':
            print("Exiting the program.")
            break

        else:
            print("Invalid option. Please choose a number between 1 and 7.")

if __name__ == "__main__":
    main()
