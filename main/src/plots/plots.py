"""Contains functions for generating plots"""

import os
import matplotlib.pyplot as plt

from typing import List


class Plots:
    """
    Class for plotting training results.
    """

    def __init__(self, directory: str, tag: str):
        """
        Initializes the plotting.
        :param directory: The directory to save the plots to.
        :param tag: An additional tag added to each plot file name.
        """
        self.directory = directory
        self.tag = tag

    def figure_path(self, file_name: str, format: str = 'png'):
        """
        Provides the path for given figure.
        :param directory: The directory to save the plot to.
        :param file_name: The name to give the file saved.
        :return: The full path to save the file as.
        """
        return os.path.join(self.directory, f'{file_name}-{self.tag}.{format}')

    def plot_performance(self, values: List[float]):
        """
        Plots the performance of a training run.
        :param values: The performance in each episode during training.
        """
        plt.figure()
        plt.plot(values)
        plt.savefig(self.figure_path('performance'))
