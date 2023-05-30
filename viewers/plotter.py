import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    def __init__(self, series):
        if not isinstance(series, pd.Series):
            raise ValueError("Input should be a pandas Series.")
        self.mean = series["mean"]
        self.sd = series["sd"]
        self.median = series["median"]
        self.IQR = series["IQR"]
        self.tenth = series["10th"]
        self.ninetieth = series["90th"]

    def display_box_plot(self):
        data = {
            "median": [self.median],
            "25th percentile": [self.median - 0.5 * self.IQR],
            "75th percentile": [self.median + 0.5 * self.IQR],
            "10th percentile": [self.tenth],
            "90th percentile": [self.ninetieth],
        }
        df = pd.DataFrame(data)

        fig, ax = plt.subplots()
        ax.boxplot(df.T, vert=True)
        ax.set_yticks(range(1, len(df.columns) + 1))
        ax.set_yticklabels(df.columns)

        plt.show()

# Example usage
series = pd.Series({
    "mean": 0.019792,
    "median": 0.000281,
    "sd": 0.087270,
    "IQR": 0.082244,
    "10th": 0.000000,
    "90th": 0.167848
})

# plotter = Plotter(series)
# plotter.display_box_plot()