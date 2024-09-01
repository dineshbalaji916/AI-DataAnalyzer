from data_processor import DataProcessor
from analysis_engine import AnalysisEngine

def display_menu():
    print("\nAI Employee Analysis Tool")
    print("=" * 40)
    print("1. Correlation Analysis")
    print("2. Clustering Analysis")
    print("3. Regression Analysis")
    print("4. Classification Analysis")
    print("5. Descriptive Statistics")
    print("6. Anomaly Detection")
    print("7. Exit")

def main():
    # Step 1: Data Ingestion
    file_path = input("Enter the file path (CSV, Excel, JSON): ")
    processor = DataProcessor()
    data = processor.load_data(file_path)
    data = processor.clean_data(data)
    processed_data = processor.preprocess_data(data)

    # Step 2: Initialize Analysis Engine
    analysis_engine = AnalysisEngine(processed_data)
    
    while True:
        display_menu()
        choice = input("Choose an analysis option (1-7): ")

        if choice == '1':
            print("Performing Correlation Analysis...")
            result = analysis_engine.correlation_analysis()
            print(result)
            insights = analysis_engine.generate_insights('correlation', result)
            print("Insights:", insights)

        elif choice == '2':
            print("Performing Clustering Analysis...")
            result, _ = analysis_engine.clustering_analysis()
            print("Clustering Results:")
            print(result)
            insights = analysis_engine.generate_insights('clustering', result)
            print("Insights:", insights)

        elif choice == '3':
            target_column = input("Enter the target column for regression: ")
            print("Performing Regression Analysis...")
            result = analysis_engine.regression_analysis(target_column)
            print(f"Coefficients: {result[0]}")
            print(f"Intercept: {result[1]}")
            insights = analysis_engine.generate_insights('regression', result)
            print("Insights:", insights)

        elif choice == '4':
            target_column = input("Enter the target column for classification: ")
            print("Performing Classification Analysis...")
            result = analysis_engine.classification_analysis(target_column)
            print(f"Feature Importances: {result}")
            insights = analysis_engine.generate_insights('classification', result)
            print("Insights:", insights)

        elif choice == '5':
            print("Calculating Descriptive Statistics...")
            result = analysis_engine.descriptive_statistics()
            print(result)
            insights = analysis_engine.generate_insights('descriptive_stats', result)
            print("Insights:", insights)

        elif choice == '6':
            print("Performing Anomaly Detection...")
            result = analysis_engine.anomaly_detection()
            print("Anomalies Detected:")
            print(result)
            insights = analysis_engine.generate_insights('anomaly_detection', result)
            print("Insights:", insights)

        elif choice == '7':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please choose a valid option.")

if __name__ == "__main__":
    main()
