# AI Analysis CLI Tool

## Overview

The AI DataAnalyzer CLI Tool is a command-line interface application designed to perform various data analysis techniques on employee data using machine learning algorithms. This tool allows users to easily ingest data, perform analyses, generate insights, and visualize results, all from the command line.

## Features

- **Correlation Analysis**: Identifies correlations between different variables in the dataset and visualizes them as a heatmap.
- **Clustering Analysis**: Groups data into clusters using K-Means clustering and visualizes the clusters.
- **Regression Analysis**: Performs linear regression to model the relationship between variables.
- **Classification Analysis**: Uses a RandomForestClassifier to identify important features for a target variable.
- **Descriptive Statistics**: Computes basic statistics for the dataset.
- **Anomaly Detection**: Identifies outliers in the dataset using an Isolation Forest model.

## Installation

1. **Clone the Repository**:

   git clone https://github.com/<your-github-username>/AI-DataAnalyzer.git
   cd AI-DataAnalyzer

2. **Install Dependencies**:

Ensure you have Python installed (version 3.6 or higher recommended). Install the required Python libraries using pip:

pip install pandas scikit-learn matplotlib seaborn

3. **Usage**

# Run the CLI Tool:

Start the application by running:

python cli.py
Follow the On-Screen Instructions:

Enter the file path for the dataset you want to analyze (supported formats: CSV, Excel, JSON).
Choose the type of analysis you want to perform from the menu.

# View Results and Insights:

Analysis results and insights will be displayed directly in the terminal.
Visualizations (e.g., heatmaps, clustering plots) will be saved in the analysis_images folder within your project directory.

4. **Project Structure**

AI-DataAnalyzer/
│
├── analysis_engine.py   # Contains the core analysis logic and methods for generating insights and visualizations
├── data_processor.py    # Handles data loading, cleaning, and preprocessing
├── cli.py               # Main command-line interface script
├── analysis_images/     # Folder where generated analysis images are saved
└── README.md            # Project documentation (you're reading it now)

# Contact

For any questions or suggestions, please reach out to me via GitHub.