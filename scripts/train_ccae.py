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

    learning_rate = 1e-5
    num_epochs = 15
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = CCAE_Dataloader(batch_size=B)

    model = CCAE(dim_input, n_objects, dim_speical_features, n_votes).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    eps=(10*B)**(-2)
                                    )
    # Start training
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            # Forward pass
            x = data['corners']
            x = x.to(device)
            res_dict = model(x)
            res_dict = ccae_loss(res_dict, x)

            loss = res_dict.log_likelihood
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()

            # seems not working
            torch.nn.utils.clip_grad_norm_(model.parameters(),True)

            optimizer.step()
            print(loss.item())


if __name__ == "__main__":
    main()