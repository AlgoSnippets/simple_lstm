import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_processor import DataProcessor


def build_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def calculate_metrics(y_true, y_pred, y_pred_binary):
    return {
        "Accuracy": accuracy_score(y_true, y_pred_binary),
        "Precision": precision_score(y_true, y_pred_binary),
        "Recall": recall_score(y_true, y_pred_binary),
        "F1 Score": f1_score(y_true, y_pred_binary),
        "Directional Accuracy": np.mean((y_pred > 0.5) == y_true)
    }

def main():
    ticker = 'AAPL'
    start_date='2020-01-01'
    end_date='2025-04-25'

    data = pd.read_csv(f"data_cache/{ticker}_data.csv", index_col=0, parse_dates=True)
    data = data.sort_index(ascending=True)
    data = data[start_date:end_date]

    # Label the data
    data['Target'] = data['Close'].shift(-1) > data['Close']
    data = data.dropna()  # Remove any NaN values

    split = int(len(data) * 0.8)
    X_train, X_test = data[:split].drop(columns=['Target']), data[split:].drop(columns=['Target'])
    y_train, y_test = data[:split]['Target'], data[split:]['Target']

    # Build sequences
    sequence_length = 20
    X_train_seq = build_sequences(X_train.values, sequence_length)
    X_test_seq = build_sequences(X_test.values, sequence_length)

    # Adjust y_train and y_test to match the sequence structure
    y_train = y_train[sequence_length:]
    y_test = y_test[sequence_length:]

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X_train_seq, y_train, epochs=30, verbose=1)

    # Predict on both train and test sets
    train_pred = model.predict(X_train_seq)
    test_pred = model.predict(X_test_seq)

    # Reshape predictions
    train_pred = train_pred.reshape(-1)
    test_pred = test_pred.reshape(-1)

    # Convert probabilities to binary predictions
    train_pred_binary = (train_pred > 0.5).astype(int)
    test_pred_binary = (test_pred > 0.5).astype(int)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, train_pred, train_pred_binary)
    test_metrics = calculate_metrics(y_test, test_pred, test_pred_binary)

    # Print metrics
    print("Training Set Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()