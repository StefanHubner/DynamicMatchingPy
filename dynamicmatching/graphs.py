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

def matched_process_plot(mu_hat, mu_star, years, cells, couples, singles):
    prh = lambda n: "$\\widehat{\\mu}_{" + n + "}$"
    prs = lambda n: "$\\mu^*_{" + n + "}$"
    cls = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

    # Split cells into two groups

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 7), dpi = 500)
    muh = mu_hat.cpu().detach().numpy()
    mus = mu_star.cpu().detach().numpy()

    def plot_cells(ax, cell_group):
        handles, labels = [], []
        for idx, k in enumerate(cell_group):
            i, j = cells[k]
            line, = ax.plot(years, muh[:,i,j], label=f'{prh(k)}',
                            color=cls[idx], linestyle='-', linewidth=0.75)
            ax.plot(years[1:], mus[1:,i,j], label=f'{prs(k)}',
                    color=cls[idx], linestyle='--', linewidth=0.75)
            ax.axvspan(2001, 2008, color='grey', alpha=0.1, linewidth=0)
            handles.append(line)
            labels.append(f'{prh(k)}')
        ax.set_ylim(bottom=0)
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

def create_heatmap_simple(tensor, labs):
    data = tensor.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(3, 3), dpi = 500)
    sns.heatmap(data, annot=True, fmt=".2f", cmap="Greys",
                ax=ax, annot_kws={'size': 8}, cbar = False)
    ax.set_xticklabels(labs)
    ax.set_yticklabels(labs)
    fig.tight_layout()
    return fig

def create_heatmap(tensor, x2lab, y2lab, ticklabs):
    data = tensor.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(3, 3), dpi=500)
    
    # Create the heatmap
    sns.heatmap(data, annot=True, fmt=".1f", cmap="Greys",
                ax=ax, annot_kws={'size': 8}, cbar=False)
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ticklabs, fontsize=8)
    
    # Set y-labels on left
    ax.set_yticklabels(ticklabs, fontsize=8)
    
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

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def plot_cf_grid(df, sex):
    cols = df.columns
    scenario_level, sex_level, est_level, state_level = range(4)
    mask = (
        cols.get_level_values(scenario_level).isin(["CFF", "CF1"])
        & (cols.get_level_values(sex_level) == sex)
        & (cols.get_level_values(est_level) == "star")
    )
    state_vals = cols[mask].get_level_values(state_level)
    states = pd.Index(state_vals).unique()
    #states = {cols[i][state_level] for i in range(len(cols)) if mask[i]}
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 12), sharex=True)
    axes = axes.ravel()
    idx = pd.IndexSlice
    for ax, state in zip(axes, states):
        sub = df.loc[:, idx[["CFF", "CF1"], sex, "star", state]].copy()
        sub.columns = sub.columns.get_level_values(0)   # -> ["CFF", "CF1"]
        sub.plot(ax=ax)
        ax.set_title(state)
        ax.set_xlabel("")
        ax.legend_.remove()
    for k in range(len(states), len(axes)):
        axes[k].set_visible(False)
    fig.tight_layout()
    return fig

def plot_estimator_grid(df: pd.DataFrame, sex: str = "M",
                        scenario: str = "CFF",
                        nrows: int = 4, ncols: int = 2):
    cols = df.columns
    scenario_level, sex_level, est_level, state_level = range(4)
    mask = (
        (cols.get_level_values(scenario_level) == scenario)
        & (cols.get_level_values(sex_level) == sex)
        & cols.get_level_values(est_level).isin(["star", "hat"])
    )
    state_vals = cols[mask].get_level_values(state_level)
    states = pd.Index(state_vals).unique()
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 12), sharex=True)
    axes = axes.ravel()
    idx = pd.IndexSlice
    for ax, state in zip(axes, states):
        sub = df.loc[:, idx[scenario, sex, ["star", "hat"], state]].copy()
        sub.columns = sub.columns.get_level_values(2)   # ["star", "hat"]
        sub.plot(ax=ax)
        ax.set_title(str(state))
        ax.set_xlabel("")
        if ax.legend_ is not None:
            ax.legend_.remove()
    for k in range(len(states), len(axes)):
        axes[k].set_visible(False)
    fig.tight_layout()
    return fig
