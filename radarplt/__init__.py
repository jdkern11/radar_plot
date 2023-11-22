from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union
import logging
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from radarplt.factory import radar_factory
from radarplt.helpers import minmax_scale

logger = logging.getLogger(__name__)


class RadarPlot:
    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        value_column: str,
        hue_column: Optional[str] = None,
        value_ranges: Optional[Dict[str, List[float]]] = {},
        plot_labels: Optional[Dict[str, str]] = {},
        target_ranges: Optional[Dict[str, List[float]]] = {},
        colors: Optional[List[str]] = [
            "#332288",
            "#117733",
            "#44AA99",
            "#88CCEE",
            "#DDCC77",
            "#CC6677",
            "#AA4499",
            "#882255",
        ],
        figsize: Optional[Tuple[float]] = (6.4, 4.8),
        target_linewidth: Optional[float] = 1
    ):
        """Creates radar plot from data in df

        Args:
            df:
                Dataframe contianing data to plot.
            label_column:
                Column in df corresponding to labels to plot
            value_column:
                Column in df corresponding to values to plot
            hue_column:
                Column in df corresponding to different objects that should be plotted
                on separate lines.
            value_ranges:
                Optional, if set, then values in value column corresponding to
                the appropriate label will be normalized to fall between 0 and 1, with
                0 being prop_ranges[prop][0] and 1 being prop_ranges[prop][1]. If it falls
                outside those ranges, a warning will be raised. If not passed, or it
                is missing for a specific label, then the min and max for the dataframe
                will be used.
            plot_labels:
                Optional, if passed, labels on the plot will be the value of the dictionary.
                If not passed or set for some label, then the label will be whatever it
                already is.
            target_ranges:
                Optional, if set, then target ranges will be added to the plot for each
                property it is defined for.
            colors:
                Colors to plot each item.
            figsize:
                Size of figure to plot
        """
        self.df = df.copy()
        self.label_column = label_column
        self.value_column = value_column
        self.hue_column = hue_column
        self.target_ranges = self._add_missing_target_ranges(target_ranges)
        self.plot_labels = self._add_missing_plot_labels(plot_labels)
        self.value_ranges = self._add_missing_value_ranges(value_ranges)
        self.colors = colors
        self.theta = radar_factory(len(self.ordered_labels), frame="polygon")
        self._add_scaled_value_column_to_df()
        self.scaled_target_ranges = self._scale_target_ranges()
        self.figsize = figsize
        self.target_linewidth = target_linewidth

    @cached_property
    def ordered_labels(self):
        self._ordered_labels = list(self.plot_labels.keys())
        self._ordered_labels.sort()
        angles = {
            self._ordered_labels[i]: i * 2 * np.pi / len(self.plot_labels)
            for i in range(len(self.plot_labels))
        }
        return self._ordered_labels

    @cached_property
    def angles(self):
        self._angles = {
            self.ordered_labels[i]: i * 2 * np.pi / len(self.plot_labels)
            for i in range(len(self.plot_labels))
        }
        return self._angles

    def _add_missing_target_ranges(
        self, target_ranges: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        self.unique_labels = list(
            set(list(self.df[self.label_column].unique()) + list(target_ranges.keys()))
        )
        if len(target_ranges) != 0:
            for label in self.unique_labels:
                if label not in target_ranges:
                    target_ranges[label] = [0, 0]
        return target_ranges

    def _add_missing_plot_labels(self, plot_labels: Dict[str, str]) -> Dict[str, str]:
        """Adds any missing labels from df to plot_labels"""
        plot_labels = copy.deepcopy(plot_labels)
        for label in self.unique_labels:
            if label not in plot_labels:
                plot_labels[label] = label
        # ignore extra labels passed to plot_labels, but not in the dataframe
        return {label: plot_labels[label] for label in self.unique_labels}

    def _add_missing_value_ranges(
        self,
        value_ranges: Dict[str, List[float]],
    ) -> Dict[str, List[float]]:
        """Adds min/max property value_ranges

        Gives warning if value ranges lower than max or higher than min in the
        dataframe.
        """
        value_ranges = copy.deepcopy(value_ranges)
        for label in self.unique_labels:
            tdf = self.df.loc[self.df[self.label_column] == label]
            if label not in value_ranges:
                value_ranges[label] = [
                    tdf[self.value_column].min(),
                    tdf[self.value_column].max(),
                ]
            else:
                if tdf[self.value_column].min() < value_ranges[label][0]:
                    warning = (
                        f"Value range min ({value_ranges[label][0]}) for "
                        + f"{label} larger than min ({tdf[self.value_column].min()}) for the df"
                    )
                    logger.warning(warning)
                if tdf[self.value_column].max() > value_ranges[label][1]:
                    warning = (
                        f"Value range max ({value_ranges[label][1]}) for "
                        + f"{label} less than max ({tdf[self.value_column].min()}) for the df"
                    )
                    logger.warning(warning)
        # ignore extra labels passed to plot_labels, but not in the dataframe
        return {label: value_ranges[label] for label in self.unique_labels}

    def _add_scaled_value_column_to_df(self) -> pd.DataFrame:
        """Scales value column in df for each property based on value_ranges"""
        scaled_values = []
        for index, row in self.df.iterrows():
            scaled_values.append(
                minmax_scale(
                    row[self.value_column], self.value_ranges[row[self.label_column]]
                )
            )
        self.df[f"_scaled_{self.value_column}"] = scaled_values

    def _scale_target_ranges(self) -> Dict[str, List[float]]:
        scaled_ranges = {}
        for key, value in self.target_ranges.items():
            scaled_ranges[key] = [
                minmax_scale(value[0], self.value_ranges[key]),
                minmax_scale(value[1], self.value_ranges[key]),
            ]
        return scaled_ranges

    def plot(self) -> Tuple[Figure, Union[List[Axes], Axes]]:
        fig, ax = plt.subplots(
            subplot_kw=dict(projection="radar"), dpi=300, figsize=self.figsize
        )
        self._plot_target_ranges(ax)
        self._plot_df(ax)
        self._add_tick_labels(fig, ax)

        return (fig, ax)

    def _plot_target_ranges(self, ax):
        if len(self.target_ranges) == 0:
            return
        first = True  # only want to plot label with first target
        for label in self.ordered_labels:
            if first:
                ax.plot(
                    [self.angles[label], self.angles[label]],
                    self.scaled_target_ranges[label],
                    c="#F95C0F",
                    label="Target",
                    linewidth=self.target_linewidth
                )
                first = False
            else:
                ax.plot(
                    [self.angles[label], self.angles[label]],
                    self.scaled_target_ranges[label],
                    c="#F95C0F",
                    linewidth=self.target_linewidth
                )

    def _plot_df(self, ax):
        curr_color = 0
        hues = self.df[self.hue_column].unique()
        for hue in hues:
            logger.info(f"Plotting {hue}")
            tdf = self.df.loc[self.df[self.hue_column] == hue]
            tdf = tdf.dropna(subset=self.value_column)
            theta, r = [], []
            for label in self.ordered_labels:
                tdf_ = tdf.loc[tdf[self.label_column] == label]
                if len(tdf_) != 0:
                    val = tdf_[f"_scaled_{self.value_column}"].to_list()[0]
                    theta.append(self.angles[label])
                    r.append(val)
            r += [r[0]]
            theta += [theta[0]]
            ax.plot(theta, r, c=self.colors[curr_color], marker="o", label=hue)
            ax.fill(theta, r, facecolor=self.colors[curr_color], alpha=0.25)
            curr_color += 1

    def _add_tick_labels(self, fig, ax):
        ax.set_rmax(1)
        ax.set_rticks(ticks=[0.25, 0.5, 0.75], labels=[])
        ax.grid(True)
        labels = [self.plot_labels[label] for label in self.ordered_labels]
        label_angles = [
            self.angles[label] * 180 / np.pi for label in self.ordered_labels
        ]

        # rotation reset to 0 for polar coordinates, so we'll remove all xticks and
        # just use the coordinates we get when we run thetagrids
        lines, labels = plt.thetagrids(label_angles, labels)
        n_labels = []
        for i in range(len(labels)):
            label = labels[i]
            x, y = label.get_position()
            lab = ax.text(
                x,
                y,
                label.get_text(),
                transform=label.get_transform(),
                ha=label.get_ha(),
                va=label.get_va(),
            )
            lab.set_rotation(label_angles[i])
            n_labels.append(lab)
        # don't want to remove the ticks, just the label text
        ax.set_xticklabels(["" for label in labels])
        labels = n_labels

        # plot ranges for values
        for i in range(len(self.ordered_labels)):
            label = labels[i]
            theta = label_angles[i] * np.pi / 180

            r = 0.25
            lower, upper = self.value_ranges[self.ordered_labels[i]]
            text = round(r * (upper - lower) + lower, 2)
            t = ax.text(
                theta + 0.2, r, text, fontsize=8, ha=label.get_ha(), va=label.get_va()
            )
            t.set_rotation(label_angles[i] + 90)

            r = 0.75
            lower, upper = self.value_ranges[self.ordered_labels[i]]
            text = round(r * (upper - lower) + lower, 2)
            t = ax.text(
                theta + 0.07, r, text, fontsize=8, ha=label.get_ha(), va=label.get_va()
            )
            t.set_rotation(label_angles[i] + 90)


def plot(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    hue_column: Optional[str] = None,
    value_ranges: Optional[Dict[str, List[float]]] = {},
    plot_labels: Optional[Dict[str, str]] = {},
    target_ranges: Optional[Dict[str, List[float]]] = {},
    figsize: Optional[Tuple[float]] = (6.4, 4.8),
    target_linewidth: Optional[float] = 1
) -> Tuple[Figure, Union[List[Axes], Axes]]:
    return RadarPlot(
        df,
        label_column,
        value_column,
        hue_column,
        value_ranges,
        plot_labels,
        target_ranges,
        figsize=figsize,
        target_linewidth=target_linewidth
    ).plot()
