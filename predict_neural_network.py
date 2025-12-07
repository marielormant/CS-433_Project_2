"""
Prediction script for the trained neural network model.
This script loads the trained model and makes predictions on new data.
"""

import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(data_path="dataset/dataset.pkl"):
    """
    Load and prepare the dataset for prediction.
    
    Args:
        data_path: Path to the pickle file containing the dataset
        
    Returns:
        X: Feature matrix (N x 22)
        y: Label vector (N,)
        feature_names: List of feature names
    """
    # Load dataset
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    
    data = pd.json_normalize(dataset)
    
    # Extract epsilon features
    eps_global = data['eps_global']
    eps_df = pd.DataFrame(
        eps_global.tolist(), 
        columns=[f"eps{i}" for i in range(1, 7)]
    )
    
    # Extract plies features
    plies_cols = [
        'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
        'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
        'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
        'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
    ]
    plies_df = data[plies_cols]
    
    # Combine features
    data_df = pd.concat([eps_df, plies_df], axis=1)
    
    # Build labels
    F = data_df[plies_cols].values
    mask_failure = (F >= 1)
    has_failure = mask_failure.any(axis=1)
    
    F_valid = F.copy()
    F_valid[F_valid < 1] = -np.inf
    
    y = np.zeros(len(F), dtype=int)
    max_FI = F_valid[has_failure].argmax(axis=1) + 1
    y[has_failure] = max_FI
    
    X = data_df.values
    feature_names = data_df.columns.tolist()
    
    return X, y, feature_names


def load_model(model_path="best_nn_model.h5"):
    """
    Load the trained neural network model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        model: Loaded Keras model
    """
    model = keras.models.load_model(model_path)
    return model


def predict(model, X, scaler=None):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained Keras model
        X: Feature matrix (N x 22)
        scaler: Optional StandardScaler for preprocessing
        
    Returns:
        predictions: Predicted class probabilities (N x 17)
        predicted_classes: Predicted class labels (N,)
    """
    # Scale features if scaler is provided
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        # If no scaler provided, assume X is already scaled
        X_scaled = X
    
    # Make predictions
    predictions = model.predict(X_scaled, batch_size=1024, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predictions, predicted_classes


def get_class_names():
    """
    Get the mapping of class indices to meaningful names.
    
    Returns:
        class_names: Dictionary mapping class indices to names
    """
    class_names = {
        0: "No Failure",
        1: "Fiber Tension (0°)",
        2: "Fiber Compression (0°)",
        3: "Matrix Tension (0°)",
        4: "Matrix Compression (0°)",
        5: "Fiber Tension (45°)",
        6: "Fiber Compression (45°)",
        7: "Matrix Tension (45°)",
        8: "Matrix Compression (45°)",
        9: "Fiber Tension (90°)",
        10: "Fiber Compression (90°)",
        11: "Matrix Tension (90°)",
        12: "Matrix Compression (90°)",
        13: "Fiber Tension (-45°)",
        14: "Fiber Compression (-45°)",
        15: "Matrix Tension (-45°)",
        16: "Matrix Compression (-45°)"
    }
    return class_names


def predict_sample(model, sample, scaler=None, verbose=True):
    """
    Predict a single sample and display results.
    
    Args:
        model: Trained Keras model
        sample: Feature vector (1 x 22)
        scaler: Optional StandardScaler for preprocessing
        verbose: Whether to print detailed results
        
    Returns:
        predicted_class: Predicted class label
        confidence: Prediction confidence (probability)
    """
    # Ensure sample is 2D
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)
    
    # Make prediction
    predictions, predicted_classes = predict(model, sample, scaler)
    
    predicted_class = predicted_classes[0]
    confidence = predictions[0][predicted_class]
    
    if verbose:
        class_names = get_class_names()
        print(f"\nPrediction Results:")
        print(f"  Predicted Class: {predicted_class} - {class_names[predicted_class]}")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Show top 3 predictions
        print(f"\n  Top 3 Predictions:")
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        for rank, idx in enumerate(top_3_idx, 1):
            prob = predictions[0][idx]
            print(f"    {rank}. Class {idx} ({class_names[idx]}): {prob:.4f} ({prob*100:.2f}%)")
    
    return predicted_class, confidence


def main():
    """
    Main function demonstrating model usage.
    """
    print("="*60)
    print("Neural Network Model - Prediction Demo")
    print("="*60)
    
    # Load model
    print("\n1. Loading trained model...")
    try:
        model = load_model()
        print("   ✓ Model loaded successfully!")
        model.summary()
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        print("   Please ensure 'best_nn_model.h5' exists in the current directory.")
        print("   Run 'python neural_network.py' first to train the model.")
        return
    
    # Load data
    print("\n2. Loading dataset...")
    try:
        X, y, feature_names = load_and_prepare_data()
        print(f"   ✓ Dataset loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return
    
    # Prepare scaler
    print("\n3. Preparing data scaler...")
    scaler = StandardScaler()
    scaler.fit(X)
    print("   ✓ Scaler fitted!")
    
    # Make predictions on a few samples
    print("\n4. Making predictions on sample data...")
    num_samples = 5
    sample_indices = np.random.choice(len(X), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n--- Sample {i} (Index: {idx}) ---")
        print(f"True Label: {y[idx]} - {get_class_names()[y[idx]]}")
        
        predicted_class, confidence = predict_sample(
            model, 
            X[idx], 
            scaler, 
            verbose=True
        )
        
        # Check if prediction is correct
        if predicted_class == y[idx]:
            print("   ✓ CORRECT prediction!")
        else:
            print("   ✗ INCORRECT prediction")
    
    print("\n" + "="*60)
    print("Prediction demo completed!")
    print("="*60)


if __name__ == "__main__":
    main()
