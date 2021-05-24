from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_bandit_agent as linear_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.utils import common