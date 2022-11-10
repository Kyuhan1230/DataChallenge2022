from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


def plot_split_indices(cv, lw=10):
    # 시각화를 위한 함수임
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    plt.style.use('seaborn-white')
    
    """Create a sample plot for indices of a cross-validation object."""
    # Get CV Splits
    n_splits = cv.n_splits
    data_size = cv.data_size
    # Visualize For Splits
    fig, ax = plt.subplots()
    plt.grid()
    for ii in range(n_splits):
        tt = cv.train_cv_indices_list[ii]
        tr = cv.test_cv_indices_list[ii]
        
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * data_size)
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Formatting
    y_tick_labels = list(range(n_splits))
    ax.set(yticks=np.arange(n_splits) + .5, yticklabels=y_tick_labels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+0.1, -.1], xlim=[0, data_size])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))], ['Training set', 'Test set'], loc=(1.02, .8))