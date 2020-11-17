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
    plt.scatter(corners[:,0], corners[:,1], c=c)
    plt.show()