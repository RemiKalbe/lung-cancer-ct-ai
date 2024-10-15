# ruff: noqa: E402

#
# Monkey 1: SafeConfigParser to ConfigParser
#
import configparser

# Monkey patch configparser.SafeConfigParser to ConfigParser
configparser.SafeConfigParser = configparser.ConfigParser  # type: ignore

import pylidc  # type: ignore # noqa: E401

#
# Monkey 2: Fixing the numpy int issue
#

import numpy as np

np.int = np.int_  # type: ignore
np.bool = bool
np.float = np.float32  # type: ignore

#
# Monkey 3: Fix bool8 issue
#

np.sctypeDict["bool8"] = np.bool_
np.dtype("bool8")  # Ensures that 'bool8' is recognized
