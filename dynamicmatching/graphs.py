import matplotlib.pyplot as plt
import numpy as np
import torch 
import seaborn as sns
import base64

def matched_process_plot(ss_hat, ss_star):
    colors = ['b', 'g', 'r', 'c', 'm']
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sh = ss_hat.cpu().detach().numpy()
    ss = ss_star.cpu().detach().numpy()
    
    for i in range(5):
        ax.plot(sh[:, i], label=f'Shat{i+1}', color=colors[i], linestyle='-')
        ax.plot(ss[:, i], label=f'Sstar{i+1}', color=colors[i], linestyle='--')
    ax.legend()
    ax.set_title('Matched Process')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    return fig

def matched_process_plot(mu_hat, mu_star, years):
    prh = lambda n: "$\\widehat{\\mu}_{" + n + "}$"
    prs = lambda n: "$\\mu^*_{" + n + "}$"
    cells = {"nn": (0,0), "ee": (1,1), "ne": (0,1), "cc": (2,2),
             "n0": (0,3), "e0": (1,3), "0n": (3,0), "0e": (3,1)}
    cls = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

    # Split cells into two groups
    couples = ["nn", "ee", "ne", "cc"]
    singles = ["n0", "e0", "0n", "0e"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 7), dpi = 500)
    muh = mu_hat.cpu().detach().numpy()
    mus = mu_star.cpu().detach().numpy()

    def plot_cells(ax, cell_group):
        handles, labels = [], []
        for idx, k in enumerate(cell_group):
            i, j = cells[k]
            line, = ax.plot(years, muh[:,i,j], label=f'{prh(k)}',
                            color=cls[idx], linestyle='-', linewidth=0.75)
            ax.plot(years, mus[:,i,j], label=f'{prs(k)}',
                    color=cls[idx], linestyle='--', linewidth=0.75)
            ax.axvspan(2001, 2008, color='grey', alpha=0.1, linewidth=0)
            handles.append(line)
            labels.append(f'{prh(k)}')
        ax.legend(handles, labels, fontsize=6,  loc='upper right')
        ax.set_xticks(years[::4])
        ax.set_xticklabels(years[::4], rotation=45, fontsize = 6)
        ax.tick_params(axis='both', which='major', labelsize=6)


    plot_cells(ax1, couples)
    plot_cells(ax2, singles)

    ax1.set_title("Couples", fontsize=8)
    ax2.set_title("Singles", fontsize=8)

    plt.tight_layout()
    return fig, ax1, ax2

def create_heatmap_simple(tensor):
    data = tensor.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(3, 3), dpi = 500)
    sns.heatmap(data, annot=True, fmt=".2f", cmap="Greys",
                ax=ax, annot_kws={'size': 8}, cbar = False)
    ax.set_xticklabels(['N', 'E', 'C', '0'])
    ax.set_yticklabels(['N', 'E', 'C', '0'])
    fig.tight_layout()
    return fig

def create_heatmap(tensor, x2lab, y2lab):
    data = tensor.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(3, 3), dpi=500)
    
    # Create the heatmap
    sns.heatmap(data, annot=True, fmt=".1f", cmap="Greys",
                ax=ax, annot_kws={'size': 8}, cbar=False)
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(['N', 'E', 'C', '0'], fontsize=8)
    
    # Set y-labels on left
    ax.set_yticklabels(['N', 'E', 'C', '0'], fontsize=8)
    
    # Create a twin of the y-axis on the right side
    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ybound(ax.get_ybound())
    ax2.set_yticklabels(y2lab[::-1], fontsize=8)
    
    # Create a twin of the x-axis at the bottom
    ax3 = ax.twiny()
    ax3.xaxis.tick_bottom()
    ax3.xaxis.set_label_position('bottom')
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xbound(ax.get_xbound())
    ax3.set_xticklabels(x2lab, fontsize=8)
    
    # Remove the original bottom axis
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    fig.tight_layout()
    return fig

def svg_to_data_url(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f'data:image/svg+xml;base64,{b64}'
