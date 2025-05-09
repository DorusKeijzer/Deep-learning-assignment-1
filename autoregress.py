import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from torch import nn
import click
from typing import Dict, Type

# Model imports
from model_architectures.lstm import LSTMModel

def load_model_from_file(filepath: str, model_classes: Dict[str, Type[nn.Module]]) -> nn.Module:
    package = torch.load(filepath, map_location="cpu")
    # FORCE CORRECT PARAMS - CRITICAL FIX
    package["model_parameters"]["input_size"] = 1  # <--- THIS WAS MISSING
    model_class = model_classes[package["model_class"]]
    model = model_class(**package["model_parameters"])
    model.load_state_dict(package["state_dict"])
    model.eval()
    return model

@click.command()
@click.option('--model_path', '-m', required=True)
@click.option('--lag_param', '-l', type=int, required=True)
def main(model_path: str, lag_param: int):
    # Load model
    model = load_model_from_file(model_path, {"LSTMModel": LSTMModel})

    # Load data
    train_data = loadmat("data/Xtrain.mat")["Xtrain"].flatten().astype(np.float32)
    test_data = loadmat("data/Xtest.mat")["Xtest"].flatten().astype(np.float32)[:200]
    
    # Initialize scaler
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()

    window = scaled_train[-lag_param:].copy()  
    predictions = []

    for _ in range(200):
        # Input tensor from SCALED window
        input_tensor = torch.tensor(window, dtype=torch.float32).view(1, lag_param, 1)
        
        # Predict
        with torch.no_grad():
            pred_scaled = model(input_tensor).item()
        
        # Store ORIGINAL-SCALE prediction
        pred = scaler.inverse_transform([[pred_scaled]]).item()
        predictions.append(pred)
        
        # Update window WITH NEW SCALED VALUE
        new_scaled = scaler.transform([[pred]]).flatten()[0] 
        window = np.concatenate([window[1:], [new_scaled]])
    # Calculate metrics
    mse = np.mean((np.array(predictions) - test_data) ** 2)
    mae = np.mean(np.abs(np.array(predictions) - test_data))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Training Data", alpha=0.7)
    plt.plot(range(len(train_data), len(train_data)+200), predictions, 
             label="Predictions", color='orange')
    plt.plot(range(len(train_data), len(train_data)+200), test_data, 
             label="True Values", color='green', linestyle='--')
    
    plt.title(f"Autoregressive Predictions\nMSE: {mse:.2f}, MAE: {mae:.2f}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
