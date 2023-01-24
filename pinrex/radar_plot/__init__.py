from typing import Dict, List, Optional, Tuple, Union
import logging
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

logger = logging.getLogger(__name__)

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    hue_column: Optional[str] = None,
    value_ranges: Optional[Dict[str, List[float]]] = {},
    plot_labels: Optional[Dict[str, str]] = {},
    target_hue: Optional[str] = None,
    greater_than_column: Optional[str] = None,
) -> Tuple[Figure, Union[List[Axes], Axes]]:
    """Creates radar plot from data in df

    Args:
        df (pd.DataFrame):
            Dataframe contianing data to plot.
        label_column (str):
            Column in df corresponding to labels to plot
        value_column (str):
            Column in df corresponding to values to plot
        hue_column (str):
            Column in df corresponding to different objects that should be plotted
            on separate lines.
        value_ranges (Optional[Dict[str, List[float]]]):
            Optional, if set, then values in value column corresponding to
            the appropriate label will be normalized to fall between 0 and 1, with
            0 being prop_ranges[prop][0] and 1 being prop_ranges[prop][1]. If it falls
            outside those ranges, a ValueError will be raised. If not passed, or it
            is missing for a specific label, then the min and max for the dataframe
            will be used.
        plot_labels (Optional[Dict[str, str]]):
            Optional, if passed, labels on the plot will be the value of the dictionary.
            If not passed or set for some label, then the label will be whatever it
            already is.
        target_hue (Optional[str]):
            Optional, if passed, then the string represents the hue that corresponds
            to the target values.
        greater_than_column (Optional[str]):
            Optional, if passed, then it is coupled with the target column to indicate
            if the target should be greater than, or less than the value.

    Returns:
        Tuple[Figure, Union[List[Axes], Axes]
    """
    plot_labels = copy.deepcopy(plot_labels)
    value_ranges = copy.deepcopy(value_ranges)
    unique_labels = df[label_column].unique()
    for label in unique_labels:
        if label not in plot_labels:
            plot_labels[label] = label

        tdf = df.loc[df[label_column] == label]
        if label not in value_ranges:
            value_ranges[label] = [tdf[value_column].min(), tdf[value_column].max()]
        else:
            if tdf[value_column].min() < value_ranges[label][0]:
                warning = (
                    f"Value range min ({value_ranges[label][0]}) for "
                    + f"{label} larger than min ({tdf[value_column].min()}) for the df"
                )
                logger.warning(warning)
                # raise ValueError(warning)
            if tdf[value_column].max() > value_ranges[label][1]:
                warning = (
                    f"Value range max ({value_ranges[label][1]}) for "
                    + f"{label} less than max ({tdf[value_column].min()}) for the df"
                )
                logger.warning(warning)
                # raise ValueError(warning)

    # remove extra labels not used
    plot_labels = {label: plot_labels[label] for label in unique_labels}
    value_ranges = {label: value_ranges[label] for label in unique_labels}

    # order labels so they are consistent when plotting
    label_order = list(plot_labels.keys())
    label_order.sort()
    angles = {
        label_order[i]: i * 2 * np.pi / len(plot_labels)
        for i in range(len(plot_labels))
    }

    hues = list(df[hue_column].unique())
    # Want to plot target hue first so it is below other items
    if target_hue is not None:
        hues.remove(target_hue)
        hues = [target_hue] + hues

    N = len(label_order)
    theta = radar_factory(N, frame='polygon')
    fig, ax = plt.subplots(subplot_kw=dict(projection='radar'), dpi=300)

    colors = ["#5DA192", "#0000A7"]
    curr_color = 0
    for hue in hues:
        logger.info(f"Plotting {hue}")
        tdf = df.loc[df[hue_column] == hue]
        tdf.dropna(subset=value_column, inplace=True)
        values = {}
        scaled = {}
        targets_ranges = {}
        for index, row in tdf.iterrows():
            scaled[row[label_column]] = (
                row[value_column] - value_ranges[row[label_column]][0]
            ) / value_ranges[row[label_column]][1]
            values[row[label_column]] = row[value_column]
            if target_hue is not None and hue == target_hue:
                if greater_than_column is not None:
                    if row[greater_than_column]:
                        targets_ranges[row[label_column]] = [
                            scaled[row[label_column]],
                            1,
                        ]
                    else:
                        targets_ranges[row[label_column]] = [
                            0,
                            scaled[row[label_column]],
                        ]
                else:
                    targets_ranges[row[label_column]] = [
                        scaled[row[label_column]],
                        scaled[row[label_column]],
                    ]
        # Plot target if needed
        if target_hue is not None and hue == target_hue:
            first = True  # only want to plot label with first target
            for label in label_order:
                if first:
                    ax.plot(
                        [angles[label], angles[label]],
                        targets_ranges[label],
                        c="#F95C0F",
                        label="Target",
                    )
                    first = False
                else:
                    ax.plot(
                        [angles[label], angles[label]],
                        targets_ranges[label],
                        c="#F95C0F",
                    )
        # Plot everything else
        else:
            theta, r = [], []
            for label in label_order:
                if label in scaled:
                    theta.append(angles[label])
                    r.append(scaled[label])
            r += [r[0]]
            theta += [theta[0]]
            ax.plot(theta, r, c=colors[curr_color], marker="o", label=hue)
            ax.fill(theta, r, facecolor=colors[curr_color], alpha=0.25)
            curr_color += 1

    ax.set_rmax(1)
    ax.set_rticks(ticks=[0.25, 0.5, 0.75], labels=[])
    ax.grid(True)
    labels = [plot_labels[label] for label in label_order]
    label_angles = [angles[label] * 180 / np.pi for label in label_order]
    lines, labels = plt.thetagrids(label_angles, labels)
    angle = np.deg2rad(67.5)
    # plot ranges for values
    for i in range(len(label_order)):
        offset = 0
        theta = label_angles[i] * np.pi / 180 + offset

        r = 0.25
        lower, upper = value_ranges[label_order[i]]
        text = round(r * upper + lower, 2)
        plt.text(theta, r, text, fontsize=8)

        r = 0.75
        lower, upper = value_ranges[label_order[i]]
        text = round(r * upper + lower, 2)
        plt.text(theta, r, text, fontsize=8)

    return (fig, ax)
