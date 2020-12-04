"""
A dirty training script
"""
import torch
from model.models.ccae import CCAE
from data_loader.ccae_dataloader import CCAE_Dataloader
from model.loss import ccae_loss
from tqdm import tqdm
from utils.plot import plot_concellation_compare


def main():
    # hyperparameters for model
    B = 4
    k = 7  # number of input set
    dim_input = 2  # 2 for 2D point
    dim_speical_features = 16
    n_votes = 4
    n_objects = 2

    # hyperparameters for training
    learning_rate = 1e-5
    num_epochs = 15
    report = 1000  # every mini-batches

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = CCAE_Dataloader(batch_size=B)

    model = CCAE(dim_input, n_objects, dim_speical_features, n_votes).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=learning_rate, momentum=0.9, eps=(10 * B) ** (-2)
    )
    # Start training
    loss_history = []
    model.train()
    im_count = 0
    for epoch in range(num_epochs):
        for i, data in tqdm(enumerate(data_loader)):
            # Forward pass
            x = data["corners"]
            x = x.to(device)
            res_dict = model(x)  # loss is computed in CCEA.forward

            loss = res_dict.log_likelihood
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), True)

            optimizer.step()
            if (i + 1) % report == 0:
                loss_history.append(loss)
                plot_concellation_compare(
                    x.cpu().numpy(),
                    data["pattern_id"].squeeze().cpu().numpy(),
                    res_dict.winners.cpu().numpy(),
                    name="saved/imgs/img_{}.jpg".format(im_count),
                )
                im_count += 1


if __name__ == "__main__":
    main()
