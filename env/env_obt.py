import logging
from PIL import Image
import itertools
import gym
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym import error, spaces
import os

class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym_unity")

config = {
    "train-mode": 1,          # Training mode enabled (1)
    "tower-seed": -1,         # Use a random tower on every reset
    "starting-floor": 0,      # Start at floor 0 (easier for early training)
    "total-floors": 10,       # Limit tower height to 10 floors (shorter episodes)
    "dense-reward": 1,        # Use dense rewards to provide more frequent feedback
    "lighting-type": 0,       # No realtime lighting for simpler visuals
    "visual-theme": 0,        # Use only the default theme for consistency
    "agent-perspective": 1,   # Third-person view (might offer a wider field of view)
    "allowed-rooms": 0,       # Only normal rooms (exclude key and puzzle rooms)
    "allowed-modules": 1,     # Only easy modules (avoid the full range of challenges)
    "allowed-floors": 0,      # Only straightforward floor layouts (no branching or circling)
    "default-theme": 0        # Set the default theme to Ancient (0)
}

class CustomObstacleTowerEnv(gym.Env):
    # If you want to keep a version check, do it in a new way or just remove it
    ALLOWED_VERSIONS = ['1', '1.1', '1.2', '1.3']

    def __init__(
        self,
        environment_filename=None,
        worker_id=43,
        mode='retro',
        timeout_wait=30,
        realtime_mode=False,
        custom_reward=False,
        config=None
    ):
        """
        This environment uses the modern ML-Agents API with no references to 'env.brains'.
        """
        self.config = config
        self.reset_parameters = EnvironmentParametersChannel()
        self.engine_config = EngineConfigurationChannel()
        if self.is_grading():
            environment_filename = None

        # 1) Create the UnityEnvironment
        self._env = UnityEnvironment(
            file_name=environment_filename,
            worker_id=worker_id,
            timeout_wait=timeout_wait,
            side_channels=[self.reset_parameters, self.engine_config]
        )

        # 2) After creation, do an initial .reset() so we can find the behavior name
        self._env.reset()

        # NEW: Pick the first (and presumably only) behavior
        self.behavior_name = list(self._env.behavior_specs.keys())[0]
        self.behavior_spec = self._env.behavior_specs[self.behavior_name]

        # [Optional] If you expect exactly one Behavior
        if len(self._env.behavior_specs) != 1:
            raise UnityGymException(
                "There can only be one behavior in this environment if it's a single-agent gym."
            )

        # If you want a version check, you can parse the behavior_name
        # e.g. "ObstacleTowerAgent-v4.1?team=0" -> version "4.1"
        # Just remove these lines if you don't need version checks.
        # self.version = "??"  # or parse from self.behavior_name
        # if self.version not in self.ALLOWED_VERSIONS:
        #     raise UnityGymException("Invalid environment version, got v" + self.version)

        self.visual_obs = None
        self._prev_key_obs = None
        self._current_state = None
        self._n_agents = 1  # Usually single agent
        self._done_grading = False
        self._flattener = None
        self._seed = None
        self._floor = None
        self.realtime_mode = realtime_mode
        self.game_over = False
        self.retro = 'retro' in mode
        self.mode = mode
        self.use_custom_reward = custom_reward
        
                
        if self.realtime_mode:
            # Realtime mode (1.0 is real time)
            self.engine_config.set_configuration_parameters(
                time_scale=3.0
            )
        else:
            self.engine_config.set_configuration_parameters(
                time_scale=10.0 # Run as fast as possible
            )
        # If the user wants 84x84 or 126x126
        flatten_branched = self.retro
        uint8_visual = self.retro
        self.uint8_visual = uint8_visual

        # 3) Build the action space from self.behavior_spec
        if len(self.behavior_spec.action_shape) == 1:
            # Discrete
            self._action_space = spaces.Discrete(self.behavior_spec.action_shape[0])
        else:
            # Multi-discrete
            if flatten_branched:
                self._flattener = ActionFlattener(self.behavior_spec.action_shape)
                self._action_space = self._flattener.action_space
            else:
                self._action_space = spaces.MultiDiscrete(self.behavior_spec.action_shape)

        # 4) For observation space: typically the first observation is the visual
        #   shape. E.g. self.behavior_spec.observation_shapes[0] might be (84,84,3).
        #   In ML-Agents, the shapes can be separate arrays in decision_steps.obs,
        #   but let's assume the first is the camera.
        if len(self.behavior_spec.observation_shapes) < 1:
            raise UnityGymException("No visual observations in the environment specs.")

        camera_shape = self.behavior_spec.observation_shapes[0]
        # e.g. camera_shape = (84,84,3). We'll override it if we're in retro mode
        camera_height, camera_width, camera_channels = camera_shape
        if self.mode == 'retro':
            camera_height, camera_width = 84, 84
        elif self.mode == 'retro_large':
            camera_height, camera_width = 126, 126
        elif self.mode == 'retro_hd':
            camera_height, camera_width = 256, 256

        depth = 3
        image_space_max = 1.0
        image_space_dtype = np.float32
        if self.retro or self.mode == 'retro_large' or self.mode =='retro_hd':
            image_space_max = 255
            image_space_dtype = np.uint8

        self._image_shape = (camera_height, camera_width, depth)
        self._image_space = spaces.Box(
            0, image_space_max,
            dtype=image_space_dtype,
            shape=self._image_shape
        )
        if self.retro:
            self._observation_space = self._image_space
        else:
            max_float = np.finfo(np.float32).max
            keys_space = spaces.Discrete(5)
            time_remaining_space = spaces.Box(low=0.0, high=max_float, shape=(1,), dtype=np.float32)
            self._observation_space = spaces.Tuple(
                (self._image_space, keys_space, time_remaining_space)
            )

        # We can do an initial get_steps to confirm single-agent
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
        # If you want to ensure single agent, do:
        n_agents = len(decision_steps) + len(terminal_steps)
        if n_agents != 1:
            raise UnityGymException("More than one agent or zero found. Single-agent environment only.")
        logger.info("%d agent(s) within environment." % n_agents)

    def done_grading(self):
        return self._done_grading

    def is_grading(self):
        return os.getenv('OTC_EVALUATION_ENABLED', False)

    def reset(self, config=None):
        """
        Gym-like reset. Call self._env.reset(), then get initial steps from .get_steps().
        Return the processed observation from that initial step.
        """
        reset_params = {}
        if self._floor is not None:
            reset_params['floor-number'] = self._floor
        if self._seed is not None:
            reset_params['tower-seed'] = self._seed
        if config is None and self.config is not None:
            config = self.config
        if config is not None:
            for key, value in config.items():
                self.reset_parameters.set_float_parameter(key, value)
        self._env.reset()
        if self.realtime_mode:
            # Realtime mode (1.0 is real time)
            self.engine_config.set_configuration_parameters(
                time_scale=3.0
            )
        else:
            self.engine_config.set_configuration_parameters(
                time_scale=10.0
            )
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
        # single agent => either 1 in decision_steps or 1 in terminal_steps
        if len(decision_steps) > 0:
            # agent is in a decision step
            agent_id = list(decision_steps.agent_id)[0]
            obs_list = decision_steps.obs
            # obs_list is a list of arrays, first is camera
            # If there's more obs, you'd parse them here
            visual = obs_list[0][agent_id]  # shape is (H, W, C) if using pytonics
            obs = self._make_observation(visual, decision_steps.reward[agent_id], False, decision_steps)
        else:
            # If the environment immediately started in terminal? Very unusual
            agent_id = list(terminal_steps.agent_id)[0]
            obs_list = terminal_steps.obs
            visual = obs_list[0][agent_id]
            obs = self._make_observation(visual, terminal_steps.reward[agent_id], True, terminal_steps)

        self.game_over = False
        self._prev_key_obs = None
        return obs

    def step(self, action):
        """
        Gym-like step.
        1) if we have an action flattener, convert action scalar -> branched array
        2) self._env.set_actions(behavior_name, the branched action)
        3) self._env.step()
        4) getSteps => either decision_steps or terminal_steps for the agent
        5) parse obs, reward, done, info
        """
        if self._flattener is not None:
            action = self._flattener.lookup_action(action)

        # Single agent => shape = (1, <action_dim>)
        action_array = np.array(action, dtype=np.float32).reshape((1, -1))
        self._env.set_actions(self.behavior_name, action_array)
        self._env.step()
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            agent_id = list(terminal_steps.agent_id)[0]
            obs_list = terminal_steps.obs
            visual = obs_list[0][agent_id]
            reward = terminal_steps.reward[agent_id]
            done = True
            obs = self._make_observation(visual, reward, done, terminal_steps)
        else:
            agent_id = list(decision_steps.agent_id)[0]
            obs_list = decision_steps.obs
            visual = obs_list[0][agent_id]
            reward = decision_steps.reward[agent_id]
            done = False
            obs = self._make_observation(visual, reward, done, decision_steps)

        # If you have text observation "evaluation_complete", you might detect that here
        # or store it in info. For example:
        info = {}
        # e.g. text obs if len(decision_steps.obs) > 1 => decision_steps.obs[1] ?

        return obs, reward, done, info

    def _make_observation(self, visual_arr, reward, done, step_result):
        """
        Convert the raw visual array and any vector data into the final Gym observation format.
        If self.retro, it's an 84x84 or 126x126 image plus possible stats overlay, etc.
        If not retro, it might be a (image, keys, time_remaining) tuple.
        Also handle custom reward if needed. (Though usually you'd do that in .step() logic.)
        """
        # For your code, you do something like self._preprocess_single(...) etc.
        if self.uint8_visual:
            raw_visual = (255.0 * visual_arr).astype(np.uint8)
        else:
            raw_visual = visual_arr

        # Step 2: HD image for video recording (must be RGB uint8)
        self.visual_obs = self._resize_observation(raw_visual, (256, 256, 3))

        # Step 3: Agent observation (downsample for model input)
        agent_obs = self._resize_observation(raw_visual, (84, 84, 3))

        return agent_obs

    def _resize_observation(self, observation, image_shape):
        h, w, c = image_shape
        if observation.shape[0] == h and observation.shape[1] == w:
            return observation
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((w, h), Image.NEAREST)
        return np.array(obs_image)

    def is_grading(self):
        return os.getenv('OTC_EVALUATION_ENABLED', False)

    def done_grading(self):
        return self._done_grading

    def close(self):
        self._env.close()

    def render(self, mode="rgb_array"):
        """
        Return the current visual observation for video recording.
        """
        if self.visual_obs is None:
            logger.warning("render() called before visual_obs is ready.")
            return np.zeros((256, 256, 3), dtype=np.uint8)

        return self.visual_obs

    # If you want a seed or floor param, store them and apply them in .reset
    def seed(self, seed=None):
        if seed is None:
            self._seed = None
        else:
            self._seed = int(seed)

    def floor(self, floor=None):
        if floor is None:
            self._floor = None
        else:
            self._floor = int(floor)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self):
        return -float('inf'), float('inf')

    @property
    def spec(self):
        return None

# We also define your ActionFlattener for branched discrete
class ActionFlattener:
    def __init__(self, branched_action_space):
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(cls, branched_action_space):
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        action_lookup = {_scalar: _action for (_scalar, _action) in enumerate(all_actions)}
        return action_lookup

    def lookup_action(self, action):
        return self.action_lookup[action]
