# Copyright (c) 2024 Mateusz Brzozowski, Bartłomiej Krawczyk, Mikołaj Kuranowski, Konrad Wojda
# SPDX-License-Identifier: MIT

# pyright: basic

import gc
import logging
import random
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from operator import attrgetter
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable, List, NamedTuple, Optional, Union, cast

import keras
import numpy as np
import numpy.typing as npt

from .environment import Action, Environment, TurtleCameraView

MODELS_DIR = Path("models")

NDArrayFloat = npt.NDArray[np.float_]

logger = logging.getLogger(__name__)


@dataclass
class DQNParameters:
    discount: float = 0.9
    """Discount for next-step reward (from the target_model).

    Tweakable, default 0.9.
    """

    initial_epsilon: float = 1.0
    """Initial epsilon - ratio of random-provided over model-provided models.

    Not tweakable.
    """

    epsilon_decay: float = 0.99
    """Epsilon decay rate. With every move, epsilon is set to `epsilon * epsilon_decay`.

    Not tweakable.
    """

    epsilon_min: float = 0.05
    """Lowest possible epsilon.

    Not tweakable.
    """

    replay_memory_max_size: int = 20_000
    """Maximum cached moves for learning.

    Tweakable, default 20 000."""

    replay_memory_min_size: int = 4_000
    """Minimum cached moves for before learning can commence.

    Tweakable, default 4 000.
    """

    minibatch_size: int = 32
    """How many moves should be randomly drawn for model learning inside a minibatch?

    Tweakable, default 32.
    """

    training_batch_divisor: int = 1
    """Used for calculating the training batch size, using the following formula:
    training_batch_size = minibatch_size // training_batch_divisor.

    Tweakable (?), default 4.
    """

    target_update_period: int = 20
    """After every target_update_period mini-batches, the target model is updated.

    Tweakable, default 20.
    """

    max_episodes: int = 4_000
    """Limit for DQN algorithm episodes.

    Not tweakable.
    """

    control_dimension: int = 6
    """Amount of possible actions.

    Technically tweakable, but limited by DQNSingle.control_to_action, default 6.
    """

    train_period: int = 4
    """After every train_period steps, the model is updated.

    Tweakable, default 4.
    """

    save_period: int = 250
    """After every save_period episodes, the model is saved to the disk.

    Not tweakable.
    """

    clear_period: int = 50
    """After every clear_period, a garbage collection is forced.

    Custom argument, tweakable.
    """

    @property
    def training_batch_size(self) -> int:
        return self.minibatch_size // self.training_batch_divisor

    def signature(self) -> str:
        return (
            f"D{self.discount}"
            f"_M{self.replay_memory_max_size}"
            f"_m{self.replay_memory_min_size}"
            f"_B{self.minibatch_size}"
            f"_U{self.target_update_period}"
            f"_T{self.train_period}"
        )


class MemoryEntry(NamedTuple):
    last_state: TurtleCameraView
    current_state: TurtleCameraView
    control: int
    reward: float
    new_state: TurtleCameraView
    done: bool


class DQNSingle:
    def __init__(
        self,
        env: Environment,
        parameters: DQNParameters = DQNParameters(),
        seed: int = 42,
        signature_prefix: str = "dqns",
    ) -> None:
        # NOTE: The following seeds Python, numpy and tensorflow RNGs
        keras.utils.set_random_seed(seed)

        self.env = env
        self.parameters = parameters
        self.model = self.make_model()
        self.target_model = self.make_model()
        self.replay_memory: "deque[MemoryEntry]" = deque(
            maxlen=self.parameters.replay_memory_max_size
        )
        self.train_count = 0
        self.epsilon = self.parameters.initial_epsilon
        self.signature_prefix = signature_prefix

    def signature(self) -> str:
        params_signature = f"{self.env.parameters.signature()}_{self.parameters.signature()}"
        params_hash = sha256(params_signature.encode("ascii")).hexdigest()[:6]
        return f"{self.signature_prefix}-{params_hash}-{params_signature}"

    @staticmethod
    def control_to_action(turtle_name: str, control: int) -> Action:
        return Action(
            turtle_name,
            speed=0.4 if control >= 3 else 0.2,
            turn=[-0.25, 0.0, 0.25][control % 3],
        )

    def input_stack(
        self,
        last: TurtleCameraView,
        current: TurtleCameraView,
        out: Optional[NDArrayFloat] = None,
    ) -> NDArrayFloat:
        if out is not None:
            assert out.shape == (self.env.parameters.grid_res, self.env.parameters.grid_res, 8)
        else:
            out = np.zeros((self.env.parameters.grid_res, self.env.parameters.grid_res, 8))

        out[:, :, 0] = current.penalties
        out[:, :, 1] = current.distances_to_goal
        out[:, :, 2] = current.speeds_along_azimuth
        out[:, :, 3] = current.speeds_perpendicular_azimuth
        out[:, :, 4] = last.penalties
        out[:, :, 5] = last.distances_to_goal
        out[:, :, 6] = last.speeds_along_azimuth
        out[:, :, 7] = last.speeds_perpendicular_azimuth

        return out

    def input_stacks(
        self,
        len: int,
        lasts: Iterable[TurtleCameraView],
        currents: Iterable[TurtleCameraView],
    ) -> NDArrayFloat:
        out = np.zeros((len, self.env.parameters.grid_res, self.env.parameters.grid_res, 8))
        for i, (last, current) in enumerate(zip(lasts, currents)):
            self.input_stack(last, current, out[i])
        return out

    def decision(
        self,
        model: keras.Sequential,
        last: TurtleCameraView,
        current: TurtleCameraView,
    ) -> NDArrayFloat:
        # Calling model (to retrieve predictions) expects a (B, n, n, m) matrix,
        # with B representing the amount of different inputs to make predictions for.
        # Since we only do a single prediction, we have to expand input_stack from (n, n, m)
        # to (1, n, n, m).
        prediction = model.predict_on_batch(
            np.expand_dims(self.input_stack(last, current), axis=0),
        )
        assert prediction.shape[0] == 1
        return prediction[0]

    def make_model(self) -> keras.Sequential:
        n = self.env.parameters.grid_res
        n = self.env.parameters.grid_res
        m = 8
        o = self.parameters.control_dimension

        # Model: "sequential"
        # ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
        # ┃ Layer (type)        ┃ Output Shape        ┃ Param # ┃
        # ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
        # │ (input)             │ (None, 5, 5, 8)     │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ reshape (Reshape)   │ (None, 5, 5, 8, 1)  │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ conv3d (Conv3D)     │ (None, 4, 4, 1, 16) │     528 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ permute (Permute)   │ (None, 4, 4, 16, 1) │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ conv3d_1 (Conv3D)   │ (None, 3, 3, 1, 16) │   1,040 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ permute_1 (Permute) │ (None, 3, 3, 16, 1) │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ conv3d_2 (Conv3D)   │ (None, 2, 2, 1, 16) │   1,040 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ flatten (Flatten)   │ (None, 64)          │       0 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ dense (Dense)       │ (None, 32)          │   2,080 │
        # ├─────────────────────┼─────────────────────┼─────────┤
        # │ dense_1 (Dense)     │ (None, 6)           │     198 │
        # └─────────────────────┴─────────────────────┴─────────┘
        #  Total params: 4,886 (19.09 KB)
        #  Trainable params: 4,886 (19.09 KB)
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
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dense(o, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        turtle_name: str,
        save_model: bool = True,
        randomize_section: bool = True,
    ) -> None:
        rewards: List[float] = []
        self.replay_memory.clear()
        self.train_count = 0
        self.epsilon = self.parameters.initial_epsilon

        for episode in range(self.parameters.max_episodes):
            start_time = perf_counter()
            reward = self.train_episode(turtle_name, randomize_section)
            rewards.append(reward)
            elapsed = perf_counter() - start_time

            logger.info(
                "Episode %d finished in %.2f s. Reward from episode %.3f, mean from last 20 %.3f.",
                episode,
                elapsed,
                reward,
                mean(rewards[-20:]),
            )

            # TODO: Studenci - okresowy zapis modelu
            if save_model and (episode + 1) % self.parameters.save_period == 0:
                self.save_model()

            if (episode + 1) % self.parameters.clear_period == 0:
                force_gc()

        self.save_model()
        force_gc()

    def train_episode(self, turtle_name: str, randomize_section: bool = True) -> float:
        self.env.reset(turtle_names=[turtle_name], randomize_section=randomize_section)
        current_state = self.env.get_turtle_camera_view(turtle_name)
        last_state = current_state.copy()
        total_reward = 0.0

        while True:
            if random.random() > self.epsilon:
                control = int(np.argmax(self.decision(self.model, last_state, current_state)))
            else:
                control = random.randint(0, self.parameters.control_dimension - 1)

            new_state, reward, done = self.env.step(
                [self.control_to_action(turtle_name, control)],
            )[turtle_name]

            total_reward += reward
            self.replay_memory.append(
                MemoryEntry(
                    last_state,
                    current_state,
                    control,
                    reward,
                    new_state,
                    done,
                )
            )

            if (
                len(self.replay_memory) >= self.parameters.replay_memory_min_size
                and self.env.step_sum % self.parameters.train_period == 0
            ):
                start_time = perf_counter()
                self.train_minibatch()
                elapsed = perf_counter() - start_time

                logger.debug("Minibatch %d finished in %.2f s", self.train_count, elapsed)
                self.train_count += 1

                if self.train_count % self.parameters.target_update_period == 0:
                    logger.debug("Updating target model weights")
                    self.target_model.set_weights(self.model.get_weights())

            if done:
                break

            last_state, current_state = current_state, new_state
            self.epsilon = max(
                self.parameters.epsilon_min,
                self.epsilon * self.parameters.epsilon_decay,
            )

        return total_reward

    def train_minibatch(self) -> None:
        moves = random.sample(self.replay_memory, self.parameters.minibatch_size)

        model_inputs = self.input_stacks(
            len=len(moves),
            lasts=map(attrgetter("last_state"), moves),
            currents=map(attrgetter("current_state"), moves),
        )

        model_expected_outputs: NDArrayFloat = self.model.predict_on_batch(model_inputs)
        target_model_next_rewards: NDArrayFloat = self.target_model.predict_on_batch(
            self.input_stacks(
                len=len(moves),
                lasts=map(attrgetter("current_state"), moves),
                currents=map(attrgetter("new_state"), moves),
            )
        )

        for idx, move in enumerate(moves):
            new_reward = move.reward
            if not move.done:
                new_reward += self.parameters.discount * np.max(target_model_next_rewards[idx])
            model_expected_outputs[idx, move.control] = new_reward

        batch_start = 0
        batch_end = self.parameters.training_batch_size
        while batch_start < self.parameters.minibatch_size:
            self.model.train_on_batch(
                x=model_inputs[batch_start:batch_end],
                y=model_expected_outputs[batch_start:batch_end],
            )
            batch_start, batch_end = batch_end, batch_end + self.parameters.training_batch_size

    def save_model(self) -> None:
        logger.debug("Saving model")
        self.model.save(MODELS_DIR / f"{self.signature()}.h5")

    def load_model(self, filename: Union[str, Path]) -> None:
        self.model = cast(keras.Sequential, keras.models.load_model(filename))
        self.target_model = keras.models.clone_model(self.model)


def force_gc() -> None:
    keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    from argparse import ArgumentParser

    import coloredlogs

    from .environment import Environment
    from .simulator import create_simulator

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    arg_parser.add_argument("-m", "--model", help="load model from this path", type=Path)
    args = arg_parser.parse_args()

    coloredlogs.install(level=logging.DEBUG if args.verbose else logging.INFO)

    with create_simulator() as simulator:
        env = Environment(simulator)
        env.setup("routes.csv", agent_limit=1)

        turtle_name = next(iter(env.agents))

        dqn = DQNSingle(env)
        if args.model:
            dqn.load_model(args.model)
        dqn.train(turtle_name, randomize_section=True)
