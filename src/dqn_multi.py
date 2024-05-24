# pyright: basic

import logging
from collections import deque
from itertools import count
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import List, Mapping, Optional, cast

import keras
import numpy as np

from .dqn_single import DQNParameters, DQNSingle, Episode, NDArrayFloat, force_gc
from .environment import Action, Environment, TurtleCameraView


class DQNMulti(DQNSingle):
    def __init__(
        self,
        env: Environment,
        parameters: DQNParameters = DQNParameters(),
        seed: int = 42,
        signature_prefix: str = "dqnm",
    ) -> None:
        super().__init__(env, parameters, seed, signature_prefix)

    def input_stack(
        self,
        last: TurtleCameraView,
        current: TurtleCameraView,
        out: Optional[NDArrayFloat] = None,
    ) -> NDArrayFloat:
        if out is not None:
            assert out.shape == (self.env.parameters.grid_res, self.env.parameters.grid_res, 10)
        else:
            out = np.zeros((self.env.parameters.grid_res, self.env.parameters.grid_res, 10))

        out[:, :, 0] = current.penalties
        out[:, :, 1] = current.distances_to_goal
        out[:, :, 2] = current.speeds_along_azimuth
        out[:, :, 3] = current.speeds_perpendicular_azimuth
        out[:, :, 4] = last.penalties
        out[:, :, 5] = last.distances_to_goal
        out[:, :, 6] = last.speeds_along_azimuth
        out[:, :, 7] = last.speeds_perpendicular_azimuth
        out[:, :, 8] = current.free_of_other_agents
        out[:, :, 9] = last.free_of_other_agents

        return out

    def make_model(self) -> keras.Sequential:
        n = self.env.parameters.grid_res
        n = self.env.parameters.grid_res
        m = 10
        o = self.parameters.control_dimension

        # Model: "sequential"
        # ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
        # ┃ Layer (type)        ┃ Output Shape        ┃ Param # ┃
        # ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
        # │ (input)             │ (None, 5, 5, 10)    │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ reshape (Reshape)   │ (None, 5, 5, 10, 1) │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ conv3d (Conv3D)     │ (None, 4, 4, 1, 20) │     820 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ permute (Permute)   │ (None, 4, 4, 20, 1) │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ conv3d_1 (Conv3D)   │ (None, 3, 3, 1, 20) │   1,620 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ permute_1 (Permute) │ (None, 3, 3, 20, 1) │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ conv3d_2 (Conv3D)   │ (None, 2, 2, 1, 20) │   1,620 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ flatten (Flatten)   │ (None, 80)          │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ dense (Dense)       │ (None, 64)          │   5,184 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ dense_1 (Dense)     │ (None, 6)           │     390 │
        # └─────────────────────┴─────────────────────┴─────────┘
        #  Total params: 9,634 (37.63 KB)
        #  Trainable params: 9,634 (37.63 KB)
        #  Non-trainable params: 0 (0.00 B)

        model = keras.Sequential()
        model.add(keras.Input(shape=(n, n, m)))
        model.add(keras.layers.Reshape(target_shape=(n, n, m, 1)))
        model.add(keras.layers.Conv3D(filters=2 * m, kernel_size=(2, 2, m), activation="relu"))
        model.add(keras.layers.Permute((1, 2, 4, 3)))
        model.add(keras.layers.Conv3D(filters=2 * m, kernel_size=(2, 2, 2 * m), activation="relu"))
        model.add(keras.layers.Permute((1, 2, 4, 3)))
        model.add(keras.layers.Conv3D(filters=2 * m, kernel_size=(2, 2, 2 * m), activation="relu"))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(o, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        return model

    def train(self, save_model: bool = True, randomize_section: bool = True) -> None:
        rewards: "deque[float]" = deque(maxlen=20)
        self.replay_memory.clear()
        self.train_count = 0
        self.episodes = 0
        self.epsilon = self.parameters.initial_epsilon

        self.env.reset(randomize_section=randomize_section)
        episodes_by_turtle = {
            name: Episode.for_agent(agent) for name, agent in self.env.agents.items()
        }

        while self.episodes < self.parameters.max_episodes:
            start_time = perf_counter()
            finished_episodes = self.train_until_first_done(episodes_by_turtle)
            elapsed = perf_counter() - start_time

            for episode in finished_episodes:
                self.env.reset_turtle(episode.turtle_name, randomize_section)
                rewards.append(episode.total_reward)
                episode.reset(self.env.agents[episode.turtle_name])
                self.increment_episode(save_model)

            self.logger.info(
                "%d episodes(s) finished (now at %d) in %.2f s. Mean reward from last 20: %.3f",
                len(finished_episodes),
                self.episodes + len(finished_episodes),
                elapsed,
                mean(rewards),
            )

        self.save_model()
        force_gc()

    def train_until_first_done(self, episodes: Mapping[str, Episode]) -> List[Episode]:
        for i in count():
            # Set controls and actions
            self.logger.debug("Ep. %03d run %02d - preparing actions", self.episodes, i)
            for name, episode in episodes.items():
                episode.control = self.get_control(episode.last_state, episode.current_state)
                episode.action = self.control_to_action(name, episode.control)

            # Execute the actions
            self.logger.debug("Ep. %03d run %02d - executing actions", self.episodes, i)
            results_by_name = self.env.step([cast(Action, i.action) for i in episodes.values()])

            # Remember action results
            for name, result in results_by_name.items():
                self.logger.debug("Ep. %03d run %02d - advancing %s", self.episodes, i, name)
                episode = episodes[name]
                episode.result = result
                self.replay_memory.append(episode.as_memory_entry())
                self.train_minibatch_if_applicable()

                episode.advance()
                self.epsilon = max(
                    self.parameters.epsilon_min,
                    self.epsilon * self.parameters.epsilon_decay,
                )

            # Check if any episode has finished
            finished_episodes = [i for i in episodes.values() if i.done]
            if finished_episodes:
                return finished_episodes

        raise RuntimeError("itertools.count() should never terminate")


if __name__ == "__main__":
    from argparse import ArgumentParser

    import coloredlogs

    from .environment import Environment
    from .simulator import create_simulator

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    arg_parser.add_argument("-m", "--model", help="load model from this path", type=Path)
    args = arg_parser.parse_args()

    with create_simulator() as simulator:
        coloredlogs.install(level=logging.DEBUG if args.verbose else logging.INFO)

        env = Environment(simulator)
        env.setup("routes.csv")
        env.parameters.detect_collisions = True

        dqn = DQNMulti(env)
        if args.model:
            dqn.load_model(args.model)
        dqn.train(randomize_section=True)
