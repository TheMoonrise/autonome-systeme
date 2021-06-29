"""Helper functions for working with the ml domain"""

import os
import random
import platform

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class Domain:
    """
    Provides utility functions to work with the unity domain.
    Functions can be platform specific.
    """

    def _build_path(self):
        """
        Provides the build path of the domain executable.
        :returns: The absolute path to the build executable and whether the environment window should be hidden.
        """
        folder, extension, hidden = None, None, False

        if (platform.system() == 'Windows'): folder, extension = 'windows', '.exe'
        if (platform.system() == 'Darwin'): folder, extension = 'mac', '.app'
        if (platform.system() == 'Linux'): folder, extension = 'linux', '.x86_64'

        if not folder: raise Exception('ML domain not configured for the current operating system')

        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, '../../../environment', folder, f'Crawler{extension}')
        return path, hidden

    def environment(self, time_scale: float = 2, quality_level: float = 1, hide_window: bool = False,
                    slipperiness: float = 0, steepness: float = 0):
        """
        Generates a new unity ml environment.
        :param time_scale: The multiplier used for the physics simulation in unity.
        If a value larger than one is used, time moves more quickly and as such the simulation.
        :param quality_level: The quality level at which the simulation if performed.
        A value below one might speed up the simulation at the cost of simulation accuracy.
        :param hide_window: When set the window is not shown and the training is performed in the backgound.
        :returns: A unity ml environment for the current platform
        """
        path, hidden = self._build_path()
        if hide_window: hidden = True

        id = random.randint(0, 65535)

        channel_eng = EngineConfigurationChannel()
        channel_env = EnvironmentParametersChannel()
        channels = [channel_eng, channel_env]

        env = UnityEnvironment(file_name=path, no_graphics=hidden, worker_id=id, side_channels=channels)

        channel_eng.set_configuration_parameters(time_scale=time_scale, quality_level=quality_level)
        channel_env.set_float_parameter('slipperiness', slipperiness)
        channel_env.set_float_parameter('steepness', steepness)

        return env
