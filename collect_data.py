
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from gym.utils.play import play


from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import gym.error
from gym import Env, logger
from gym.core import ActType, ObsType
from gym.error import DependencyNotInstalled
from gym.logger import deprecation
import pygame
from pygame import Surface
from pygame.event import Event
from pygame.locals import VIDEORESIZE

# Get action and state at line 103

class Collector:
    def __init__(self, task):
        self.learning_rate = 5
        self.gamma = .9
        self.action = random.randint(0, 3)
        self.reps = 600
        self.task = task
    
    def collect(self):

        mapping = {(pygame.K_LEFT, pygame.K_UP,): np.array([1, -1]), (pygame.K_RIGHT, pygame.K_UP,): np.array([1, 1]), 
                    (pygame.K_LEFT,): np.array([0, -1]), (pygame.K_UP,): np.array([1, 0]), (pygame.K_RIGHT,): np.array([0, 1])                     
        }
        self.play(keys_to_action=mapping, seed=42, noop=np.array([0,0]))
    
    def play(self, 
            transpose: Optional[bool] = True,
            fps: Optional[int] = None,
            zoom: Optional[float] = None,
            callback: Optional[Callable] = None,
            keys_to_action: Optional[Dict[Union[Tuple[Union[str, int]], str], ActType]] = None,
            seed: Optional[int] = None,
            noop: ActType = 0,
        ):
        env = gym.make(
            self.task, 
            render_mode="rgb_array_list",
            gravity = -4.0,
            continuous = True
            )

        env.reset(seed=seed)

        if keys_to_action is None:
            if hasattr(env, "get_keys_to_action"):
                keys_to_action = env.get_keys_to_action()
            elif hasattr(env.unwrapped, "get_keys_to_action"):
                keys_to_action = env.unwrapped.get_keys_to_action()
            else:
                raise Exception(
                    f"{env.spec.id} does not have explicit key to action mapping, "
                    "please specify one manually"
                )
        assert keys_to_action is not None

        key_code_to_action = {}
        for key_combination, action in keys_to_action.items():
            key_code = tuple(
                sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
            )
            key_code_to_action[key_code] = action

        game = PlayableGame(env, key_code_to_action, zoom)

        if fps is None:
            fps = env.metadata.get("render_fps", 30)

        done, obs = True, None
        clock = pygame.time.Clock()

        while game.running:
            if done:
                done = False
                obs = env.reset(seed=seed)
            else:
                action = key_code_to_action.get(tuple(sorted(game.pressed_keys)), noop)
                prev_obs = obs
                obs, rew, terminated, truncated, info = env.step(action)
                # GET ACTION AND STATE HERE
                print(action)
                print(rew)
                done = terminated or truncated
                if callback is not None:
                    callback(prev_obs, obs, action, rew, terminated, truncated, info)
            if obs is not None:
                rendered = env.render()
                if isinstance(rendered, List):
                    rendered = rendered[-1]
                assert rendered is not None and isinstance(rendered, np.ndarray)
                self.display_arr(
                    game.screen, rendered, transpose=transpose, video_size=game.video_size
                )

            # process pygame events
            for event in pygame.event.get():
                game.process_event(event)

            pygame.display.flip()
            clock.tick(fps)
        pygame.quit()
    
    def display_arr(self, 
        screen: Surface, arr: np.ndarray, video_size: Tuple[int, int], transpose: bool
    ):
        """Displays a numpy array on screen.
        Args:
            screen: The screen to show the array on
            arr: The array to show
            video_size: The video size of the screen
            transpose: If to transpose the array on the screen
        """
        arr_min, arr_max = np.min(arr), np.max(arr)
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, video_size)
        screen.blit(pyg_img, (0, 0))


class PlayableGame:
    """Wraps an environment allowing keyboard inputs to interact with the environment."""

    def __init__(
        self,
        env: Env,
        keys_to_action: Optional[Dict[Tuple[int, ...], int]] = None,
        zoom: Optional[float] = None,
    ):
        """Wraps an environment with a dictionary of keyboard buttons to action and if to zoom in on the environment.
        Args:
            env: The environment to play
            keys_to_action: The dictionary of keyboard tuples and action value
            zoom: If to zoom in on the environment render
        """
        if env.render_mode not in {"rgb_array", "rgb_array_list"}:
            logger.error(
                "PlayableGame wrapper works only with rgb_array and rgb_array_list render modes, "
                f"but your environment render_mode = {env.render_mode}."
            )

        self.env = env
        self.relevant_keys = self._get_relevant_keys(keys_to_action)
        self.video_size = self._get_video_size(zoom)
        self.screen = pygame.display.set_mode(self.video_size)
        self.pressed_keys = []
        self.running = True

    def _get_relevant_keys(
        self, keys_to_action: Optional[Dict[Tuple[int], int]] = None
    ) -> set:
        if keys_to_action is None:
            if hasattr(self.env, "get_keys_to_action"):
                keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, "get_keys_to_action"):
                keys_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                raise Exception(
                    f"{self.env.spec.id} does not have explicit key to action mapping, "
                    "please specify one manually"
                )
        assert isinstance(keys_to_action, dict)
        relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
        return relevant_keys

    def _get_video_size(self, zoom: Optional[float] = None) -> Tuple[int, int]:
        rendered = self.env.render()
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        video_size = (rendered.shape[1], rendered.shape[0])

        if zoom is not None:
            video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))

        return video_size

    def process_event(self, event: Event):
        """Processes a PyGame event.
        In particular, this function is used to keep track of which buttons are currently pressed
        and to exit the :func:`play` function when the PyGame window is closed.
        Args:
            event: The event to process
        """
        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            self.running = False
        elif event.type == VIDEORESIZE:
            self.video_size = event.size
            self.screen = pygame.display.set_mode(self.video_size)



if __name__=="__main__":
    if len(sys.argv) > 1:
        task = sys.argv[1]
    else:
        task = "LunarLander-v2"
    collector = Collector(task)
    collector.collect()