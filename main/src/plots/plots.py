"""Contains functions for generating plots"""

import os
from types import LambdaType
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Union


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

    def plot_performance(self, values: Union[List[float], List[List[float]]], title: str):
        """
        Plots the performance of a training run.
        :param values: The performance in each episode during training.
        """
        plt.style.use('ggplot')

        if not isinstance(values[0], list):
            values = [values]
        plt.figure(figsize=self.figsize)
        for v in values:
            plt.plot(v)
        plt.title(title)
        plt.savefig(self.figure_path(title.lower().replace(' ', '-')))

    def plot_moving_avg_performance(self, values: List[float], title: str, window_size: int = 100):
        """
        Plots the moving average of the performance of the training run.
        :param values: The performance of each episode during training.
        :param window_size: The number of samples to consider for calculating the moving average.
        """
        # change plot style
        plt.style.use('ggplot')

        # calculate moving average
        window = np.ones(int(window_size)) / float(window_size)
        moving_avg = np.convolve(values, window, 'valid')

        plt.figure(figsize=self.figsize)

        # plot data
        plt.plot(values, color='royalblue', alpha=0.3, label='Performance')
        plt.plot(moving_avg, color='royalblue', label='Moving average of performance')

        # set plot information
        plt.title(title)
        plt.ylabel('Performance')
        plt.legend()

        # save plot
        plt.savefig(self.figure_path(f'avg_{title}'))

    def plot_moving_avg_std_performance(self, values: List[float], title: str, window_size: int = 100):

        np_values = np.array(values)

        # calculate moving average
        window = np.ones(int(window_size)) / float(window_size)
        moving_avg = np.convolve(values, window, 'valid')

        # calculate moving standard deviation
        cumsum = np.cumsum(np_values)
        cumsum_square = np.cumsum(np_values**2)
        cumsum = np.insert(cumsum, 0, 0)                    # Insert a 0 at the beginning of the array
        cumsum_square = np.insert(cumsum_square, 0, 0)      # Insert a 0 at the beginning of the array
        seg_sum = cumsum[window_size:] - cumsum[:-window_size]
        seg_sum_square = cumsum_square[window_size:] - cumsum_square[:-window_size]
        std = np.sqrt(seg_sum_square / window_size - (seg_sum / window_size)**2)
        std = std.tolist()

        plt.figure(figsize=self.figsize)

        # plot data
        plt.plot(moving_avg, color='royalblue', label='Moving average of performance')
        plt.fill_between(x=np.arange(len(moving_avg)), y1=moving_avg + std, y2=moving_avg - std, alpha=0.3,
                         facecolor='royalblue', label='Moving standard deviation of performance')

        # set plot information
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.legend()

        # save plot
        plt.savefig(self.figure_path(f'avg_std_{title}'))
