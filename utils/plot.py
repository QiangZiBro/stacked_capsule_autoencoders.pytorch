import matplotlib.pyplot as plt

_COLORS = """
    #a6cee3
    #1f78b4
    #b2df8a
    #33a02c
    #fb9a99
    #e31a1c
    #fdbf6f
    #ff7f00
    #cab2d6
    #6a3d9a
    #ffff99
    #b15928""".split()


def plot_concellation(data_dict):
    corners = data_dict.corners
    pattern_id = data_dict.pattern_id.squeeze()
    c = [_COLORS[i] for i in pattern_id]
    plt.scatter(corners[:, 0], corners[:, 1], c=c)
    plt.show()


def plot_concellation_compare(x, y, y_pre, name=None):
    """
    batch plot for original data and predicted data(label is unsupervisedly classified)
    Args:
        x: (B,M,2)
        y: (B,M)
        y_pre: (B,M)

    Returns:

    """
    B, M = y.shape
    assert x.shape[-1] == 2
    plt.clf()
    fig, axs = plt.subplots(2, B, figsize=(15, 15))
    for i in range(B):
        axs[0, i].scatter(x[i, :, 0], x[i, :, 1], c=[_COLORS[j] for j in y[i]])
        axs[1, i].scatter(x[i, :, 0], x[i, :, 1], c=[_COLORS[j] for j in y_pre[i]])

    if name:
        plt.savefig(name)
    else:
        plt.show()
