import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl   
import torch.optim as optim
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer    
import requests
import zipfile

# Constants
tot_action_classes = 6
data_dir = "RNN-HAR-2D-Pose-database" 
data_file_name = "ntu_data.zip"
file_id = "1IuZlyNjg6DMQE3iaO1Px6h1yLKgatynt"

# Dataset Class
class ActionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Model Definition
class ActionClassificationLSTM(pl.LightningModule):
    def __init__(self, input_features, hidden_dim, num_classes=6, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_features, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.train_losses = []
        self.train_accs = []

    def forward(self, x):
        lstm_out, (ht, _) = self.lstm(x)
        return self.linear(ht[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y).long()
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        pred = y_pred.argmax(dim=1)
        acc = torchmetrics.functional.accuracy(
            pred, y, task="multiclass", num_classes=tot_action_classes
        )
        self.train_losses.append(loss.detach().clone().cpu())
        self.train_accs.append(acc.detach().clone().cpu())
        self.log('batch_train_loss', loss, prog_bar=True)
        self.log('batch_train_acc', acc, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.train_losses and self.train_accs:
            avg_loss = torch.stack(self.train_losses).mean()
            avg_acc = torch.stack(self.train_accs).mean()
            self.log('train_loss', avg_loss, prog_bar=True)
            self.log('train_acc', avg_acc, prog_bar=True)
        self.train_losses.clear()
        self.train_accs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self(x)
            return torch.argmax(output, dim=1)

# Function to download file from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/u/1/uc?id=1IuZlyNjg6DMQE3iaO1Px6h1yLKgatynt"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Main execution
if __name__ == "__main__":
    # Download and extract dataset if not present
    if not os.path.exists(os.path.join(data_dir, 'X_train.txt')):
        if not os.path.exists(data_file_name):
            print("Downloading dataset...")
            download_file_from_google_drive(file_id, data_file_name)
            print("Download complete.")
        print(f"Extracting data from {data_file_name}...")
        try:
            with zipfile.ZipFile(data_file_name, 'r') as zip_ref:
                zip_ref.extractall('.')  # Extract to current directory
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print(f"Error: {data_file_name} is not a valid zip file. Please check the downloaded file.")
            exit()
    else:
        print("Dataset already extracted.")

    # Load data
    try:
        X_train = np.loadtxt(os.path.join(data_dir, 'X_train.txt'), delimiter=',')
        y_train = np.loadtxt(os.path.join(data_dir, 'Y_train.txt'), delimiter=',')
        val_exists = (
            os.path.exists(os.path.join(data_dir, 'X_val.txt')) and
            os.path.exists(os.path.join(data_dir, 'Y_val.txt'))
        )
        if val_exists:
            X_val = np.loadtxt(os.path.join(data_dir, 'X_val.txt'), delimiter=',')
            y_val = np.loadtxt(os.path.join(data_dir, 'Y_val.txt'), delimiter=',')
        else:
            X_val, y_val = None, None
            print("Validation files not found. Skipping validation.")
    except Exception as e:
        print(f"Error loading data from text files: {e}")
        exit()

    print(f"Loaded X_train shape: {X_train.shape}")
    print(f"Loaded y_train shape: {y_train.shape}")

    # Infer sequence_length from data
    total_samples = X_train.shape[0]
    total_labels = y_train.shape[0]
    num_features_per_frame = X_train.shape[1]  # 36

    # Calculate sequence_length
    if total_samples % total_labels == 0:
        sequence_length = total_samples // total_labels
        print(f"Inferred sequence_length: {sequence_length}")
        X_train = X_train.reshape(-1, sequence_length, num_features_per_frame)
    else:
        print("Error: Cannot infer sequence_length. Check your data shapes.")
        exit()

    print("After reshape:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Reshape y if necessary
    if y_train.ndim > 1:
        y_train = y_train.squeeze()
    if y_val is not None and y_val.ndim > 1:
        y_val = y_val.squeeze()

    # Ensure labels are in the range 0 to tot_action_classes-1
    print("Unique labels before correction:", np.unique(y_train))
    if y_train.min() == 1 and y_train.max() == tot_action_classes:
        print("Remapping labels from 1-based to 0-based indexing.")
        y_train -= 1
        if y_val is not None:
            y_val -= 1
    print("Unique labels after correction:", np.unique(y_train))

    input_features = num_features_per_frame
    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final y_train shape: {y_train.shape}")

    # Create datasets and dataloaders
    train_dataset = ActionDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=3)
    # No validation loader since files are missing

    hidden_dim = 128
    model = ActionClassificationLSTM(input_features=input_features, hidden_dim=hidden_dim)

    # Only monitor train_loss since no validation
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints/',
        filename='best-model',
        save_top_k=1,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    trainer.fit(model, train_loader)

    trainer.save_checkpoint("final_model.ckpt")
    print("Model training complete. Checkpoint saved as final_model.ckpt.")
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    model = ActionClassificationLSTM.load_from_checkpoint(best_model_path)