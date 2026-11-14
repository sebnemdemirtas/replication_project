# REPLICATION CODE FOR STAT 5200 by Sebnem Demirtas

from pathlib import Path
import pandas as pd
import statsmodels.api as sm

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
# Run the replication`  `
########################################################################

project_dir = Path(__file__).resolve().parent.parent
data = load_data(project_dir)
make_table2(data, project_dir / "results" / "tables" / "table2.csv")
