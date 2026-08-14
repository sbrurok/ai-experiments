from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_data():

    # Training data
    train_data_label = np.load('data/train_data.npz')['data']
    train_data_label = torch.tensor(train_data_label, dtype=torch.float32)

    train_data_input = train_data_label.clone()
    train_data_input[:, :, 10:18, 10:18] = 0

    # Test data
    test_data_input = np.load('data/test_data.npz')['data']
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)

    # Normalization
    train_data_label = train_data_label / 255.0
    train_data_input = train_data_input / 255.0
    test_data_input = test_data_input / 255.0
  
    return train_data_input, train_data_label, test_data_input

def training(train_data_input, train_data_label):
    model = Model()
    model.train()
    model.to(device)

    criterion = nn.MSELoss() #######
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #######

    batch_size = 128 ######
    dataset = TensorDataset(train_data_input, train_data_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_epochs = 10 #######

    for epoch in range(n_epochs):
        for x, y in tqdm(
            data_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            # loss = criterion(output, y)
            loss = criterion(output[:, :, 10:18, 10:18], y[:, :, 10:18, 10:18])
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} loss: {loss.item()}")

    return model

def testing(model, test_data_input):
    
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        test_data_output = []
        batch_size = 64

        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i : i + batch_size])
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    test_data_output = test_data_output.numpy() * 255.0
    save_data_clipped = np.clip(test_data_output, 0, 255)
    save_data_uint8 = save_data_clipped.astype(np.uint8)

    # For submission
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    filepath = f'submissions/submission_{Path(__file__).stem}.npz'
    np.savez_compressed(file=filepath, data=save_data)

def visualize(filepath):

    input_data = np.load("data/test_data.npz")["data"]
    output_data = np.load(filepath)["data"]

    reconstructed_data = input_data.copy()
    reconstructed_data[:, :, 10:18, 10:18] = output_data[:, :, 10:18, 10:18]

    start = 70
    num_images = 20

    fig, axs = plt.subplots(nrows=2, ncols=num_images, figsize=(2 * num_images, 4))

    for col, i in enumerate(range(start, start + num_images)):
        input_img = input_data[i].squeeze()
        output_img = reconstructed_data[i].squeeze()

        axs[0, col].imshow(input_img, cmap="gray")
        axs[0, col].set_title(f"Input {i}")
        axs[0, col].axis("off")

        axs[1, col].imshow(output_img, cmap="gray")
        axs[1, col].set_title(f"Output {i}")
        axs[1, col].axis("off")

    plt.tight_layout()
    plt.show(block=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) # 128, 7, 7

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    
def main():
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data_input, train_data_label, test_data_input = load_data()

    model = training(train_data_input, train_data_label)

    testing(model, test_data_input)

    visualize(filepath='submissions/submission_v1.npz')

    return None

if __name__ == "__main__":
    main()
