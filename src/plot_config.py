import matplotlib.pyplot as plt
import seaborn as sns

PUSHBACK_COLORS = {
    "weak": "#2ecc71",
    "medium": "#f39c12",
    "adversarial": "#e74c3c",
}

STAT_COLORS = {
    "accuracy": "#2980b9",
    "extracted_rate": "#8e44ad",
}

TABULATE_CONFIG = {
    "headers": "firstrow",
    "tablefmt": "fancy_outline",
    "floatfmt": ".1f",
}


LINE_STYLE = {
    "fmt": "-o",
    "linewidth": 2,
    "zorder": 4,
}

BAR_STYLE = {
    "error_kw": {"elinewidth": 1.5, "ecolor": "#333333"},
    "zorder": 3,
}


def apply_plot_config() -> None:
    """
    Apply global matplotlib and seaborn styles.
    Call this function at the top of any plotting functions.
    """

    sns.set_theme(style="whitegrid", context="talk")

    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # Typography
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.titlepad": 20,
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.labelpad": 15,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            # Line & Marker Defaults
            "lines.linewidth": 3,
            "lines.markersize": 9,
            # Error Bar Defaults
            "errorbar.capsize": 5,  # Applies capsize globally
        }
    )
