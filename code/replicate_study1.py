# REPLICATION CODE FOR STAT 5200 by Sebnem Demirtas

from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import t

########################################################################
# Load the authors' Study 1 data
########################################################################

def load_data(project_dir):
    data_file = project_dir / "data" / "study1" / "study1_dtpanel.csv"
    columns = ["userid", "searchtask", "rank", "cond", "clicked",
               "clickorder", "bonusA", "bonusB"]
    return pd.read_csv(data_file, usecols=columns)


########################################################################
# Table 2: Learning to find higher Bonus B
########################################################################

def make_table2(data, output_file):
    first_clicks = data.loc[data["clickorder"] == 1].copy() #We look at first looked option.
    first_clicks["Task x Random"] = (first_clicks["searchtask"] * (first_clicks["cond"] == "random"))
    first_clicks["Task x Strong"] = (first_clicks["searchtask"] * (first_clicks["cond"] == "strong"))
    first_clicks = first_clicks.rename(columns={"searchtask": "Task"})
    columns = ["bonusB", "Task", "Task x Random", "Task x Strong"]
    values = first_clicks[columns]
    means = values.groupby(first_clicks["userid"]).transform("mean")
    within = values - means # Subtracting each participant's means to include participant fixed effects.

    # Model is an OLS:
    model = sm.OLS(within["bonusB"], within[columns[1:]]).fit(
        cov_type="cluster",
        cov_kwds={"groups": first_clicks["userid"]},
        use_t=True,
    )

    # Save as csv
    table = pd.DataFrame({"Coefficient": model.params,
        "Std. Error": model.bse,
        "p-value": model.pvalues})
    table.to_csv(output_file, index_label="Term")
    total_variation = ((values["bonusB"] - values["bonusB"].mean()) ** 2).sum()
    overall_r2 = 1 - model.ssr / total_variation

    # Save as txt
    summary = "\n".join([
        "Table 2. Learning to Find Higher Bonus B - Replication",
        table.to_string(float_format=lambda value: f"{value:.4f}"),
        f"Observations: {int(model.nobs):,}",
        f"Participant fixed effects: {first_clicks['userid'].nunique():,}",
        f"R-squared: {overall_r2:.5f}",
        f"Within R-squared: {model.rsquared:.5f}",
        "Standard errors clustered by participant.",
    ])
    print(summary)
    output_file.with_suffix(".txt").write_text(summary + "\n", encoding="utf-8")

########################################################################
# Helper functions
########################################################################

def rank_averages(task_data, outcome):
    group = ["cond", "rank"]
    data = task_data[group + ["userid", outcome]].dropna().copy()
    grouped = data.groupby(group)[outcome]
    results = grouped.mean().to_frame("average")
    counts = grouped.size()

    # Sum deviations by participant for clustered standard errors.
    data["deviation"] = data[outcome] - grouped.transform("mean")
    cluster_sums = data.groupby(group + ["userid"])["deviation"].sum()
    squared_sums = cluster_sums.pow(2).groupby(level=group).sum()
    n = len(data)
    clusters = data["userid"].nunique()
    correction = (clusters / (clusters - 1)) * ((n - 1) / (n - len(results)))
    standard_error = (correction * squared_sums).pow(0.5) / counts
    margin = t.ppf(0.975, df=clusters - 1) * standard_error
    results["lower"] = results["average"] - margin
    results["upper"] = results["average"] + margin
    return results.reset_index().rename(columns={"cond": "condition"})



def plot_rank_averages(ax, results):
    conditions = [
        ("random", "Random", "o", -0.18),
        ("random_informed", "Random-Informed", "s", 0),
        ("strong", "Strong", "^", 0.18),
    ]
    # Offset the markers slightly so the confidence intervals stay visible.
    for condition, label, marker, offset in conditions:
        points = results.loc[results["condition"] == condition].sort_values("rank")
        lower_error = points["average"] - points["lower"]
        upper_error = points["upper"] - points["average"]
        ax.errorbar(
            points["rank"] + offset,
            points["average"],
            yerr=[lower_error, upper_error],
            fmt=marker,
            linestyle="none",
            linewidth=1,
            capsize=2,
            label=label,
        )
    ax.set_xticks(range(1, 11))
    ax.set_axisbelow(True)
    ax.grid(axis="both", which="both", color="#dddddd", linewidth=0.6)
    ax.tick_params(axis="y", which="minor", length=0)

########################################################################
# Figure 3
########################################################################

def make_figure3(data, output_file):
    task_groups = [("Task 1", 1, 1), ("Tasks 2–5", 2, 5), ("Tasks 6–10", 6, 10)]
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

    for ax, (title, start, end) in zip(axes, task_groups):
        task_data = data.loc[data["searchtask"].between(start, end)]
        plot_rank_averages(ax, rank_averages(task_data, "clicked"))
        ax.set_title(title)
        ax.set_ylabel("Pr(Search)")
        ax.yaxis.set_major_locator(MultipleLocator(0.10))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    axes[0].legend()
    axes[-1].set_xlabel("Rank")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

########################################################################
# Run the replication
########################################################################

project_dir = Path(__file__).resolve().parent.parent
data = load_data(project_dir)
make_table2(data, project_dir / "results" / "tables" / "table2.csv")
make_figure3(data, project_dir / "results" / "figures" / "figure3.png")
