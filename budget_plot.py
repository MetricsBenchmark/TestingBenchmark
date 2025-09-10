import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

'''
RQ2. Heatmap plot.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# ---------- Font Settings ----------
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False

# ---------- Load combined results ----------
df = df_all

# Replace '-' with NaN
df.replace('-', np.nan, inplace=True)

# Convert numeric columns to float safely
numeric_cols = ['failures', 'acc_improvement', 'clusters', 'rmse']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Criterion directions: ascending=False means higher is better
criterion_directions = {
    'failures': False,
    'clusters': False,
    'rmse': True,  # smallest rmse value gets rank = 1.0
    'acc_improvement': False  # largest acc improvement gets rank = 1.0
}

import pandas as pd

# Your mapping
metric_map = {
    'rnd': 'Rand',
    'gini': 'Gini',
    'ent': 'Ent',
    'nac': 'NC',
    'kmnc': 'KMNC',
    'gd': 'GD',
    'std': 'STD',
    'lsa': 'LSA',
    'dsa': 'DSA',
    'ces': 'CES',
    'pace': 'PACE',
    'est': 'EST',
    'dr': 'DR',
    'mcp': 'MCP',
    'dat': 'DAT'
}
criteria_map = {
    'failures': '#Mis.',
    'acc_improvement': 'Acc.%',
    'clusters': '#Clu.',
    'rmse': 'AE%'
}
metric_order = ["Rand", "Gini", "Ent", "NC", "KMNC", "GD", "STD", "LSA", "DSA", "CES", "PACE", "EST", "DR", "MCP",
                "DAT"]
# Assuming 'selection_metric' column exists
df['selection_metric'] = df['selection_metric'].map(metric_map).fillna(df['selection_metric'])

# Ensure final order for plotting
df['selection_metric'] = pd.Categorical(df['selection_metric'], categories=metric_map.values(), ordered=True)
df = df.sort_values('selection_metric')

# ---------- Plot heatmaps for each criterion ----------
fig, axes = plt.subplots(1, 4, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1.25]})

axes = axes.flatten()

for idx, (criterion, asc) in enumerate(criterion_directions.items()):
    ax = axes[idx]

    # Compute ranking
    ranking_df = (
        df.groupby(['budget', 'selection_metric'])[criterion].mean().reset_index()
    )
    ranking_df['rank'] = ranking_df.groupby('budget')[criterion].rank(ascending=asc, method='min')

    # Pivot for heatmap
    heatmap_data = ranking_df.pivot(index='selection_metric', columns='budget', values='rank')
    heatmap_data = heatmap_data.reindex(metric_order)

    # Plot
    if idx == 0:
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm_r", ax=ax,
                    annot_kws={"size": 16, "weight": "bold"}, cbar=False)  # cbar_kws={'label': 'Average Rank'}

    elif idx == 3:
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm_r", ax=ax,
                    annot_kws={"size": 16, "weight": "bold"}, cbar=True)
        ax.set_yticklabels([])
    else:
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm_r", ax=ax,
                    annot_kws={"size": 16, "weight": "bold"}, cbar=False)
        ax.set_yticklabels([])

    ax.set_xlabel("Budget", fontsize=20)
    ax.set_title(f"{criteria_map[criterion]}", fontsize=25)
    ax.set_ylabel("", fontsize=20)  # Selection Metric
    # Increase font size for x-axis tick labels (budgets)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    # Increase font size for y-axis tick labels (selection metrics)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

plt.tight_layout()
plt.savefig('./heatmap_ranking_new.png', dpi=800)