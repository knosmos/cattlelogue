import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from cattlelogue.unet import UNet
from cattlelogue.datasets import build_dataset, load_rf_results

import os
import numpy as np
import click

@click.command()
@click.option(
    "--epochs", type=int, default=50, help="Number of training epochs"
)
@click.option(
    "--batch_size", type=int, default=16, help="Size of each training batch"
)
@click.option(
    "--learning_rate", type=float, default=0.001, help="Initial learning rate"
)
@click.option(
    "--step_size", type=int, default=10, help="Step size for learning rate scheduler"
)
@click.option(
    "--gamma", type=float, default=0.1, help="Multiplicative factor for learning rate decay"
)
def train_unet_model(
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    step_size=10,
    gamma=0.1,
):
    """
    Train a U-Net model for livestock density prediction.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Initial learning rate for the optimizer.
        step_size (int): Step size for the learning rate scheduler.
        gamma (float): Multiplicative factor for learning rate decay.
    """
    # Load dataset
    dataset = build_dataset(process_ee=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, optimizer, and scheduler
    model = UNet(in_channels=3, out_channels=1).cuda()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.cuda(), masks.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}")

    print("Training complete.")
    torch.save(model.state_dict(), "unet_model.pth")

if __name__ == "__main__":
    train_unet_model()