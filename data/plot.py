import os

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from pinrex.db_models import Polymer, PolymerProperty, Property

from pinrex import radar_plot

engine = create_engine(
    "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        os.environ.get("PINREX_DB_USER"),
        os.environ.get("PINREX_DB_PASSWORD"),
        os.environ.get("PINREX_DB_HOST"),
        os.environ.get("PINREX_DB_PORT"),
        os.environ.get("PINREX_DB_NAME"),
    )
)

query = """
SELECT polymers.smiles AS application, polymers.id, polymers.category, properties.name,
    properties.short_name AS property, polymer_properties.value AS value, 
    polymer_properties.error_value
FROM polymers
INNER JOIN polymer_properties ON polymers.id = polymer_properties.pol_id
INNER JOIN properties ON polymer_properties.property_id = properties.id
WHERE
polymers.category = 'virtual_forward_synthesis'
"""
polymers = pd.read_sql(query, engine)
polymers.rename(
    columns={"application": "Application", "value": "Value", "property": "Property"},
    inplace=True,
)

screening = pd.read_csv("data.csv")
df = pd.read_csv("exp_values.csv")

props = {
    "Tg": [0, 900],
    "Tm": [0, 900],
    "Tc": [0, 900],
    "Td": [0, 900],
    "TS_b": [0, 150],
    "eps_b": [0, 350],
    "E": [0, 4],
    "perm_O2": [0, 1],
    "perm_CO2": [0, 1],
    "Cp": [0, 3],
    "rho": [0, 1.5],
    "TS_y": [0, 100],
}
prop_labels = {
    "Tg": "$T_{g}$",
    "Tm": "$T_{m}$",
    "Tc": "$T_{c}$",
    "Td": "$T_{d}$",
    "TS_b": "$\sigma_{b}$",
    "eps_b": "$\epsilon_{b}$",
    "E": "E",
    "perm_O2": "$\mu_{O_{2}}$",
    "perm_CO2": "$\mu_{CO_{2}}$",
    "Cp": "$C_{p}$",
    "rho": "$\\rho$",
    "TS_y": "$\sigma_{y}$",
}

rows = []
for index, row in df.iterrows():
    for col in df.columns:
        if col not in [
            "Polymer",
            "pol_id",
            " Abb.",
            "applications",
            "smiles",
            "%-usage",
        ]:
            rows.append(
                {"Application": row["Abb."], "Property": col, "Value": row[col]}
            )
df = pd.DataFrame(rows)

df = pd.concat([screening, df, polymers])
for polymer in polymers.Application.unique():
    for application in screening.Application.unique():
        tdf = df.loc[df.Application.isin([application, polymer])]
        tdf = tdf.loc[
            tdf.Property.isin(
                tdf.loc[
                    tdf.Application == application
                ].Property.to_list()
            )
        ]
        fig, ax = radar_plot.plot(
            tdf,
            label_column="Property",
            value_column="Value",
            hue_column="Application",
            value_ranges=props,
            plot_labels=prop_labels,
            target_hue=application,
            greater_than_column="Greater_Than",
        )
        plt.tight_layout()
        save_name = f"{application}_{polymer}"
        fig.savefig(f"{save_name}.png", dpi=300)
        legend = ax.legend(loc=(0.9, .95))
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax.axis('off')
        fig.savefig(f"{save_name}_legend.png", dpi=300, bbox_inches=bbox)
        plt.close("all")
        exit()
