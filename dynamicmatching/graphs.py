from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


SCENARIO_COLOURS = {"CFF": "#C44E52", "CF0": "#C44E52", "CF1": "#55A868"}


def matched_process_plot(mu_hat, mu_star, years, cells, couples, singles,
                         policy_years=(2001, 2008)):
    prh = lambda n: "$\\widehat{\\mu}_{" + n + "}$"
    prs = lambda n: "$\\mu^*_{" + n + "}$"
    cls = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

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
            handles.append(line)
            labels.append(f'{prh(k)}')
        ax.axvspan(*policy_years, color='grey', alpha=0.1, linewidth=0)
        ax.set_ylim(bottom=0)
        ax.legend(handles, labels, fontsize=6,  loc='upper right')
        ax.set_xticks(years[::4])
        ax.set_xticklabels(years[::4], rotation=45, fontsize = 6)
        ax.tick_params(axis='both', which='major', labelsize=6)


    plot_cells(ax1, couples)
    plot_cells(ax2, singles)

    ax1.set_title("Couples", fontsize=8)
    ax2.set_title("Singles", fontsize=8)

    fig.tight_layout()
    return fig, ax1, ax2

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

def _select(df, **levels):
    selectors = tuple(levels.get(name, slice(None)) for name in df.columns.names)
    return df.loc[:, selectors]


def _plot_series(ax, frame, level, beyond=0, forecast_start=None):
    sub = frame.copy()
    sub.columns = sub.columns.get_level_values(level)
    colours = [SCENARIO_COLOURS[col] for col in sub.columns] if level == "scenario" else None
    cut = len(sub) - beyond if forecast_start is None else sub.index.astype(int).searchsorted(forecast_start)

    if cut < len(sub):
        for col, colour in zip(sub.columns, colours):
            ax.plot(sub.index[:cut], sub[col].iloc[:cut], '-',
                    color=colour, linewidth=2, label=col)
            ax.plot(sub.index[cut-1:], sub[col].iloc[cut-1:], '--',
                    color=colour, linewidth=2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    else:
        sub.plot(ax=ax, color=colours)


def _plot_grid(frame, panels, draw, dim, figsize, sharex=True, sharey=False, pymin=None):
    fig, axes = plt.subplots(*dim, figsize=figsize, sharex=sharex,
                             sharey=sharey, squeeze=False)
    axes = axes.ravel()
    panels = list(panels)
    for ax, (title, selectors) in zip(axes, panels):
        draw(ax, _select(frame, **selectors))
        ax.set_title(str(title))
        ax.set_xlabel("")
        if ax.legend_ is not None:
            ax.legend_.remove()

    visible = axes[:len(panels)]
    if len(visible):
        if pymin is not None:
            ymax = max(ax.get_ylim()[1] for ax in visible)
            for ax in visible:
                ax.set_ylim(pymin, ymax)
        else:
            max_range = max(ax.get_ylim()[1] - ax.get_ylim()[0] for ax in visible)
            for ax in visible:
                ymin, ymax = ax.get_ylim()
                mid = (ymin + ymax) / 2
                ax.set_ylim(mid - max_range / 2, mid + max_range / 2)

    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_cf_grid(df, sex, dim=(4, 2), beyond=0, pymin=0,
                 forecast_start=None):
    selected = _select(df, scenario=["CFF", "CF1"], sex=sex, estimator="star")
    states = selected.columns.get_level_values("state").unique()
    panels = ((state, {"state": state}) for state in states)
    nrows, ncols = dim
    figsize = (10, 12) if (nrows > 1 and ncols > 1) else (10, 4)
    # 2026-10-12: Share grid formatting and the solid/forecast line renderer.
    draw = partial(_plot_series, level="scenario", beyond=beyond, forecast_start=forecast_start)
    fig = _plot_grid(selected, panels, draw, dim, figsize, sharex=(nrows > 1), pymin=pymin)
    return fig, selected


def plot_estimator_grid(df: pd.DataFrame, sex: str = "M",
                        scenario: str = "CFF",
                        nrows: int = 4, ncols: int = 2):
    selected = _select(df, scenario=scenario, sex=sex, estimator=["star", "hat"])
    states = selected.columns.get_level_values("state").unique()
    panels = ((state, {"state": state}) for state in states)
    fig = _plot_grid(selected, panels, partial(_plot_series, level="estimator"),
                     (nrows, ncols), (10, 12), sharey=True)
    return fig, selected


def plot_margin_counterfactuals(df: pd.DataFrame,
                                estimator: str = "star",
                                scenarios=("CF0", "CF1"),
                                beyond: int = 0,
                                forecast_start=None):
    states = ["Unmarried", "Home Production", "Market Work"]
    selected = _select(df, scenario=list(scenarios), sex=["M", "F"],
                       estimator=estimator, state=states)
    panels = ((f"{sex}, {state}", {"sex": short, "state": state})
              for sex, short in [("Male", "M"), ("Female", "F")] for state in states)
    draw = partial(_plot_series, level="scenario", beyond=beyond, forecast_start=forecast_start)
    fig = _plot_grid(selected, panels, draw, (2, 3), (12, 6), pymin=0)
    return fig, selected
