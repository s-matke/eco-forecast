import pandas as pd
import argparse
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def load_data(file_path):
    # TODO: Load processed data from CSV file

    df = pd.read_csv(file_path)

    print(df.head(2))
    print(df.columns)
    print(df.shape)
    print("-" * 10)

    df.drop('StartTime', axis=1, inplace=True)
    
    test_set = df.tail(int(0.2 * df.shape[0]))
    test_set.to_csv('data/test_data.csv', index=False)
    df.drop(test_set.index, inplace=True)

    print(df.shape)
    
    return df

def split_data(df):
    # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)

    # Save the test.csv file for predicting later
    X = df.drop(['label'], axis=1)
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    # TODO: Initialize your model and train it
    print("Training the model...")
    params = {
        'activation': 'relu', 
        'batch_size': 80, 
        'early_stopping': True, 
        'hidden_layer_sizes': (64, 64, 64), 
        'learning_rate': 'constant', 
        'learning_rate_init': 0.001, 
        'max_iter': 250, 
        'n_iter_no_change': 15, 
        'random_state': 42, 
        'shuffle': True, 
        'solver': 'adam', 
        'validation_fraction': 0.2,
        'verbose': False
    }

    model = MLPClassifier(**params).fit(X_train, y_train)
    
    return model

def save_model(model, model_path):
    # TODO: Save your trained model
    print(f"Saving the model at {model_path}")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)