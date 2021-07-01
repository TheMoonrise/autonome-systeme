"""Contains functions for generating plots"""

import os
import matplotlib.pyplot as plt
import numpy as np

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
        self.figsize = [15, 6]

    def figure_path(self, file_name: str, format: str = 'png'):
        """
        Provides the path for given figure.
        :param directory: The directory to save the plot to.
        :param file_name: The name to give the file saved.
        :return: The full path to save the file as.
        """
        return os.path.join(self.directory, f'{file_name}-{self.tag}.{format}')

    def plot_performance(self, values: List[float], title: str):
        """
        Plots the performance of a training run.
        :param values: The performance in each episode during training.
        """
        plt.figure(figsize=self.figsize)
        plt.plot(values)
        plt.title(title)
        plt.savefig(self.figure_path(title.lower().replace(' ', '-')))

    def plot_moving_avg_performance(self, values: List[float], window_size: int = 100):
        """
        Plots the moving average of the performance of the training run.
        :param values: The performance of each episode during training.
        :param window_size: The number of samples to consider for calculating the moving average.
        """
        # change plot style
        # plt.style.use('ggplot')

        # calculate moving average
        window = np.ones(int(window_size)) / float(window_size)
        moving_avg = np.convolve(values, window, 'valid')

        plt.figure(figsize=self.figsize)

        # plot data
        plt.plot(values, color='royalblue', alpha=0.3, label='Performance')
        plt.plot(moving_avg, color='royalblue', label='Moving average of performance')

        # set plot information
        # plt.title('Moving Average of Performance')
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        plt.legend()

        # save plot
        plt.savefig(self.figure_path('avg-performance'))
