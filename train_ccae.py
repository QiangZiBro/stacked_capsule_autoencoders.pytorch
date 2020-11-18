"""
A dirty training script
"""
import torch
from model.models.ccae import CCAE
from data_loader.ccae_dataloader import CCAE_Dataloader
from model.loss import ccae_loss

def main():
    B = 4
    k = 7  # number of input set
    dim_input = 2  # 2 for 2D point
    dim_speical_features = 16
    n_votes = 4
    n_objects = 3

    learning_rate = 1e-3
    num_epochs = 15
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = CCAE_Dataloader(batch_size=B)

    model = CCAE(dim_input, n_objects, dim_speical_features, n_votes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Start training
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            # Forward pass
            x = data['corners']
            x = x.to(device)
            res_dict = model(x)
            print(res_dict)
            loss = ccae_loss(res_dict, x)

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())


if __name__ == "__main__":
    main()