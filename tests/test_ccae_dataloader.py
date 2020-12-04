from data_loader.ccae_dataloader import CCAE_Dataset
from utils.plot import plot_concellation


def test_ccae_dataset():
    dataset = CCAE_Dataset()  # (which_patterns="all")
    plot_concellation(dataset[0])
