from tqdm import tqdm
import os
import datetime
from utils.dataloader import create_complete_loader
from autoregress import load_model_from_file
import click
from model_architectures.lstm import LSTMModel
import torch
from torch import nn

def train_loop(model: LSTMModel,
               train_loader, 
               learning_rate: float,
               loss_fn,
               optimizer: torch.optim.Optimizer) -> float:

    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        # FIXED: Proper reshaping to [batch_size, seq_len, input_size]
        x = x.unsqueeze(-1)  # Changed from unsqueeze(1) to unsqueeze(-1)
        y = y.unsqueeze(-1)  # Keep as [batch_size, 1]

        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples

LSTMs = [
    (LSTMModel, {"hidden_dim": 64, "num_layers": 2}),
    (LSTMModel, {"hidden_dim": 64, "num_layers": 3}),
    (LSTMModel, {"hidden_dim": 128, "num_layers": 3}),
]

LOSS_FUNC = nn.MSELoss()

@click.command()
@click.option("--lag_param", type=int, required=True)
@click.option("--model_path", type=str, required=False)
def main(lag_param, model_path):
    test_loader, test_scaler = create_complete_loader("data/Xtest.mat", lag_param, 32)
    train_loader, train_scaler = create_complete_loader("data/Xtrain.mat", lag_param, 32)

    epochs = 500
    early_stop_patience = 10
    lr_sched_patience = 5
    min_delta = 0.0001

    for model_class, model_params in tqdm(LSTMs):
        # FIXED: Correct model initialization
        model = LSTMModel(
            input_size=1,  # WAS: input_size=lag_param
            hidden_dim=model_params["hidden_dim"],
            num_layers=model_params["num_layers"],
            output_size=1
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=lr_sched_patience
        )

        best_scaled_mse = float('inf')
        epochs_no_improve = 0
        best_state = None
        scaled_train_mse = []

        print(f"\nTraining model: {model_class.__name__} with params {model_params}")

        for epoch in range(1, epochs + 1):
            scaled_mse_train = train_loop(model, train_loader, 1e-3, loss_fn=LOSS_FUNC, optimizer=optimizer)
            scaled_train_mse.append(scaled_mse_train)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} - Scaled MSE: {scaled_mse_train:.6f} - LR: {current_lr:.6f}")

            scheduler.step(scaled_mse_train)

            if scaled_mse_train < best_scaled_mse - min_delta:
                best_scaled_mse = scaled_mse_train
                epochs_no_improve = 0
                best_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        # Save model
        timestamp = datetime.datetime.now().strftime("%H_%M_%S")
        os.makedirs("final_models", exist_ok=True)
        save_path = os.path.join("final_models", f"model_{timestamp}.pt")
        torch.save({
            "model_class": model.__class__.__name__,
            "model_parameters": model_params,
            "state_dict": best_state
        }, save_path)
        print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()
