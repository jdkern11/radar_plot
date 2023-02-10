from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import radarplt

# see tables above
df = pd.read_csv("example_data.csv")
fig, ax = radarplt.plot(
    df,
    label_column="property",
    value_column="value",
    hue_column="item",
)

folder = Path.cwd() / ".." / "images"
save = str(folder / "example_1.png")
legend = ax.legend(loc=(0.9, 0.95))
plt.tight_layout()
plt.savefig(save)
