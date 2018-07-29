from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import cv2
import gym
import gym.spaces as spaces
import retro
import numpy as np
import time
import os
import csv

class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers. This object should only be
        converted to numpy array before being passed to the model. You'd not
        believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack the k last frames.

        Returns a lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(shp[0], shp[1], shp[2] * k),
                                            dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 80
        self.height = 80
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1),
                                            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        # buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT",
        #            "C", "Y", "X", "Z"]
        # actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'],
        #            ['DOWN'], ['DOWN', 'B'], ['B']]
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        actions = [['LEFT'], ['RIGHT'], ['B'], ['L'], ['R']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO. This is incredibly important
    and effects performance a lot.
    """
    def reward(self, reward):
        return reward * 0.01

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done:
                break
        return ob, totrew, done, info


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done:
                break
        return ob, totrew, done, info


class Monitor(gym.Wrapper):
    def __init__(self, env, monitorfile, logfile=None):
        gym.Wrapper.__init__(self, env)
        self.file = open(monitorfile, 'w')
        self.csv = csv.DictWriter(self.file, ['r', 'l', 't'])
        #if logfile is not None:
        self.log = open(logfile, 'w')
        self.logcsv = csv.DictWriter(self.log, ['l', 't'])
        self.episode_reward = 0
        self.episode_length = 0
        self.total_length = 0
        self.start = None
        self.csv.writeheader()
        self.file.flush()
        #if logfile is not None:
        self.logcsv.writeheader()
        self.log.flush()
        self.logfile = logfile

    def reset(self, **kwargs):
        if not self.start:
            self.start = time.time()
        else:
            self.csv.writerow({
                'r': self.episode_reward,
                'l': self.episode_length,
                't': time.time() - self.start
            })
            self.file.flush()
        self.episode_length = 0
        self.episode_reward = 0
        return self.env.reset(**kwargs)

    def step(self, ac):
        ob, rew, done, info = self.env.step(ac)
        self.episode_length += 1
        self.total_length += 1
        self.episode_reward += rew
        #if self.logfile is not None:
        if self.total_length % 1000 == 0:
            self.logcsv.writerow({
                'l': self.total_length,
                't': time.time() - self.start
            })
            self.log.flush()
        return ob, rew, done, info

    def __del__(self):
        self.file.close()


def make(game, state, stack=True, scale_rew=True, monitordir='logs/', bk2dir='videos/'):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make(game, state)
    if bk2dir:
        env.auto_record('videos/')
    if monitordir:
        #env = Monitor(env, os.path.join(monitordir, 'monitor.csv'), os.path.join(monitordir, 'log.csv'))
        time_int = int(time.time())
        env = Monitor(env, os.path.join('monitor_{}.csv'.format(time_int)), os.path.join('log_{}.csv'.format(time_int)))

    env = StochasticFrameSkip(env, n=6, stickprob=0.0)

    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env
