# REPLICATION CODE FOR STAT 5200 by Sebnem Demirtas

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
# Run the replication
########################################################################

project_dir = Path(__file__).resolve().parent.parent
data = load_data(project_dir)