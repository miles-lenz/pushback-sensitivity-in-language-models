import matplotlib.pyplot as plt
import seaborn as sns

PUSHBACK_COLORS = {
    "weak": "#2ecc71",
    "medium": "#f39c12",
    "adversarial": "#e74c3c",
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
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.titlepad": 20,
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.labelpad": 15,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "lines.linewidth": 3,
            "lines.markersize": 9,
        }
    )
