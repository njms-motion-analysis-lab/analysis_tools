import matplotlib.pyplot as plt
import pandas as pd
from viewers.plotter import Plotter
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import plotly.graph_objects as go


class MultiPlotter:
    def __init__(self, plotters):
        if not all(isinstance(plotter, Plotter) for plotter in plotters):
            raise ValueError("All input objects should be instances of the Plotter class.")
        self.plotters = plotters

    def display_combined_box_plot(self, labels=None, title=None):
        data = []

        for i, plotter in enumerate(self.plotters):
            data.append([
                plotter.tenth,  # 10th percentile
                plotter.median - 0.5 * plotter.IQR,  # 25th percentile
                plotter.median,  # Median
                plotter.median + 0.5 * plotter.IQR,  # 75th percentile
                plotter.ninetieth,  # 90th percentile
            ])
        
        # Define boxplot properties for customization
        medianprops = dict(linestyle='-', linewidth=1, color='darkgoldenrod')
        boxprops = dict(linestyle='-', linewidth=1.5, color='firebrick')
        whiskerprops = dict(linestyle='-', linewidth=1, color='black')
        capprops = dict(linestyle='-', linewidth=1, color='black')
        flierprops = dict(marker='o', markersize=5, linestyle='none')

        fig, ax = plt.subplots()
        # Create the box plot
        ax.boxplot(data, vert=True, patch_artist=True, boxprops=boxprops,
                   medianprops=medianprops, whiskerprops=whiskerprops,
                   capprops=capprops, flierprops=flierprops)
        
        # Set the labels and title
        ax.set_xticklabels(labels if labels else [f"Plotter {i+1}" for i in range(len(self.plotters))])

        plt.xticks(rotation=90)  # Add this line to rotate x-axis labels
        if title:
            plt.title(title)

        # Create a list of patches for the legend
        legend_patches = [
            mpatches.Patch(color='darkgoldenrod', label='Median'),
            mpatches.Patch(color='firebrick', label='Box: 25th to 75th Percentile (IQR)'),
            Line2D([0], [0], color='black', label='Whiskers: 10th to 90th Percentile'),
            Line2D([0], [0], color='black', linestyle='none', marker='o', markersize=5, label='Outliers')
        ]

        # Add the legend to the plot
        ax.legend(handles=legend_patches)

        # Show the plot
        plt.show()

    def create_combined_box_plot(self, labels=None, title=None):
        data = []

        for i, plotter in enumerate(self.plotters):
            data.append([
                plotter.tenth,  # 10th percentile
                plotter.median - 0.5 * plotter.IQR,  # 25th percentile
                plotter.median,  # Median
                plotter.median + 0.5 * plotter.IQR,  # 75th percentile
                plotter.ninetieth,  # 90th percentile
            ])
        
        # Define boxplot properties for customization
        medianprops = dict(linestyle='-', linewidth=1, color='darkgoldenrod')
        boxprops = dict(linestyle='-', linewidth=1.5, color='firebrick')
        whiskerprops = dict(linestyle='-', linewidth=1, color='black')
        capprops = dict(linestyle='-', linewidth=1, color='black')
        flierprops = dict(marker='o', markersize=5, linestyle='none')

        fig, ax = plt.subplots()
        # Create the box plot
        ax.boxplot(data, vert=True, patch_artist=True, boxprops=boxprops,
                medianprops=medianprops, whiskerprops=whiskerprops,
                capprops=capprops, flierprops=flierprops)
        
        # Set the labels and title
        ax.set_xticklabels(labels if labels else [f"Plotter {i+1}" for i in range(len(self.plotters))])

        plt.xticks(rotation=90)  # Add this line to rotate x-axis labels
        if title:
            plt.title(title)

        # Create a list of patches for the legend
        legend_patches = [
            mpatches.Patch(color='darkgoldenrod', label='Median'),
            mpatches.Patch(color='firebrick', label='Box: 25th to 75th Percentile (IQR)'),
            Line2D([0], [0], color='black', label='Whiskers: 10th to 90th Percentile'),
            Line2D([0], [0], color='black', linestyle='none', marker='o', markersize=5, label='Outliers')
        ]

        # Add the legend to the plot
        ax.legend(handles=legend_patches)

        # Return the figure instead of showing it
        return fig

    def create_combined_box_plot_py(self, labels=None, title=None):
        data = []

        for i, plotter in enumerate(self.plotters):
            data.append(go.Box(
                y=[
                    plotter.tenth,  # 10th percentile
                    plotter.median - 0.5 * plotter.IQR,  # 25th percentile
                    plotter.median,  # Median
                    plotter.median + 0.5 * plotter.IQR,  # 75th percentile
                    plotter.ninetieth,  # 90th percentile
                ],
                name=labels[i] if labels else f"Plotter {i+1}",
                boxpoints='all',  # display the original data points
                jitter=0.3,  # spread them out so they're more visible
                pointpos=-1.8  # position of the data points
            ))
        
        layout = go.Layout(
            title=title,
            showlegend=False
        )

        fig = go.Figure(data=data, layout=layout)
        return fig

    def save_boxplot_as_html(plotter, labels, title, filename):
        data = []

        for i, plotter in enumerate(plotter.plotters):
            data.append(go.Box(
                y=[
                    plotter.tenth,  # 10th percentile
                    plotter.median - 0.5 * plotter.IQR,  # 25th percentile
                    plotter.median,  # Median
                    plotter.median + 0.5 * plotter.IQR,  # 75th percentile
                    plotter.ninetieth,  # 90th percentile
                ],
                name=labels[i] if labels else f"Plotter {i+1}",
                boxpoints='all',  # display the original data points
                jitter=0.3,  # spread them out so they're more visible
                pointpos=-1.8  # position of the data points
            ))
        
        layout = go.Layout(
            title=title,
            showlegend=False
        )

        fig = go.Figure(data=data, layout=layout)

        py.plot(fig, filename=filename)

    def combined_stats(self):
        all_means = []
        all_medians = []
        all_sds = []
        all_IQRs = []
        all_tenths = []
        all_ninetieths = []

        for plotter in self.plotters:
            all_means.append(plotter.mean)
            all_medians.append(plotter.median)
            all_sds.append(plotter.sd)
            all_IQRs.append(plotter.IQR)
            all_tenths.append(plotter.tenth)
            all_ninetieths.append(plotter.ninetieth)

        combined_stats = pd.Series({
            'mean': np.mean(all_means),
            'median': np.mean(all_medians),
            'sd': np.mean(all_sds),
            'IQR': np.mean(all_IQRs),
            '10th': np.mean(all_tenths),
            '90th': np.mean(all_ninetieths)
        })

        return combined_stats

    @classmethod
    def convert_feature_name(cls, feature_name):
        # Remove the 'grad_data__' prefix if present
        if feature_name.startswith("grad_data__"):
            feature_name = feature_name.replace("grad_data__", "")
        
        
        feature_name = feature_name.replace("_", " ")

        return feature_name.title()


