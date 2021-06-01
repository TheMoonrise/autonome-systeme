"""Helper functions for working with the ml domain"""

import os
import platform

from mlagents_envs.environment import UnityEnvironment


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
        if (platform.system() == 'Windows'): folder, extension = "windows", "exe"

        if not folder: raise Exception('ML domain not configured for the current operating system')

        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, '../../../environment', folder, f'Crawler.{extension}')
        return path, hidden

    def environment(self):
        """
        Generates a new unity ml environment.
        :returns: A unity ml environment for the current platform
        """
        path, hidden = self._build_path()
        return UnityEnvironment(file_name=path, no_graphics=hidden)
