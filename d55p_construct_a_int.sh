#!/bin/bash

# Load libraries and dependencies
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_analysis

# Define constants
MODEL_DIR="models"
DATA_DIR="data"
RESULTS_DIR="results"

# Function to load dataset
load_data() {
  python -c "import pandas as pd; pd.read_csv('$DATA_DIR/dataset.csv').to_csv('$DATA_DIR/dataset_processed.csv', index=False)"
}

# Function to train machine learning model
train_model() {
  python -c "from sklearn.ensemble import RandomForestClassifier; from sklearn.model_selection import train_test_split; 
    from sklearn.metrics import accuracy_score, classification_report;

    X = pd.read_csv('$DATA_DIR/datasetProcessed.csv').drop(['target'], axis=1)
    y = pd.read_csv('$DATA_DIR/datasetProcessed.csv')['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    import pickle
    with open('$MODEL_DIR/model.pkl', 'wb') as f:
      pickle.dump(clf, f)"
}

# Function to analyze model
analyze_model() {
  python -c "import pickle; 
    with open('$MODEL_DIR/model.pkl', 'rb') as f:
      clf = pickle.load(f)

    importances = clf.feature_importances_
    print('Feature Importances:')
    for i, importance in enumerate(importances):
      print(f'Feature {i+1}: {importance:.3f}')"
}

# Main script
echo "Loading dataset..."
load_data

echo "Training model..."
train_model

echo "Analyzing model..."
analyze_model

echo "Results saved to $RESULTS_DIR"