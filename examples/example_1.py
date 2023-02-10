from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from pinrex import radar_plot

# see tables above
df = pd.read_csv('example_data.csv')
fig, ax = radar_plot.plot(
    df,
    label_column="property",
    value_column="value",
    hue_column="item",
)

folder = Path.cwd() / '..' / 'images'
save = str(folder / 'example_1.png')
plt.savefig(save)

