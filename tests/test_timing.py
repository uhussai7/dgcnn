import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import timeit
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Define a simple model
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(10, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)

# Create some random data
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32)

# Initialize the model
model = SimpleModel()

# Timing callback
class TimingCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Start the timer at the beginning of each epoch
        self.epoch_start_time = timeit.default_timer()

    def on_train_epoch_end(self, trainer, pl_module):
        # End the timer at the end of each epoch and calculate the elapsed time
        epoch_end_time = timeit.default_timer()
        epoch_time = epoch_end_time - self.epoch_start_time
        print(f"Epoch {trainer.current_epoch + 1} time: {epoch_time:.4f} seconds")

# Train the model
trainer = pl.Trainer(max_epochs=5, callbacks=[TimingCallback()])
trainer.fit(model, train_loader)
