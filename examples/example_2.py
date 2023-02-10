from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from pinrex import radar_plot

value_ranges = {
  "prop1": [0, 20],
  "prop2": [0, 5],
  "prop3": [0, 50],
}

# see tables above
df = pd.read_csv('example_data.csv')
fig, ax = radar_plot.plot(
    df,
    label_column="property",
    value_column="value",
    hue_column="item",
    value_ranges=value_ranges,
)

folder = Path.cwd() / '..' / 'images'
save = str(folder / 'example_2.png')
plt.savefig(save)

