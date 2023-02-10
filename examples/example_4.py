from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from pinrex import radar_plot

target_ranges = {"prop1": [10, 20], "prop2": [0, 2], "prop3": [25, 35]}

value_ranges = {
    "prop1": [0, 20],
    "prop2": [0, 5],
    "prop3": [0, 50],
}

plot_labels = {"prop1": "$\sigma^{2}$", "prop2": "Property 2 (seconds)", "prop3": "p3"}

# see tables above
df = pd.read_csv("example_data.csv")
fig, ax = radar_plot.plot(
    df,
    label_column="property",
    value_column="value",
    hue_column="item",
    value_ranges=value_ranges,
    plot_labels=plot_labels,
    target_ranges=target_ranges,
)
folder = Path.cwd() / ".." / "images"
save = str(folder / "example_4.png")
legend = ax.legend(loc=(0.9, 0.95))
plt.tight_layout()
plt.savefig(save)
