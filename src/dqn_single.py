from collections import deque
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, List, NamedTuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore
from keras.layers import Conv3D, Dense, Flatten, Permute  # type: ignore
from keras.models import Sequential  # type: ignore
from tensorflow import keras  # type: ignore

from .env_base import Action, EnvBase, TurtleCameraView

MODELS_DIR = Path("models")

NDArrayFloat = npt.NDArray[np.float_]


@dataclass(frozen=True)
class DQNParameters:
    discount: float = 0.8
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.05
    replay_memory_max_size: int = 20_000
    replay_memory_min_size: int = 1_500
    minibatch_size: int = 32

    training_batch_divisor: int = 4
    """Used for calculating the training batch size, using the following formula:
    training_batch_size = minibatch_size // training_batch_divisor
    """

    target_update_period: int = 20
    max_episodes: int = 4_000
    control_dimension: int = 10
    train_period: int = 4
    save_period: int = 200

    @property
    def training_batch_size(self) -> int:
        return self.minibatch_size // self.training_batch_divisor


class MemoryEntry(NamedTuple):
    last_state: TurtleCameraView
    current_state: TurtleCameraView
    control: int
    reward: float
    new_state: TurtleCameraView
    done: bool


class DQNSingle:
    def __init__(
        self, env: EnvBase, parameters: DQNParameters = DQNParameters(), seed: int = 42
    ) -> None:
        self.random = Random(seed)
        self.env = env
        self.parameters = parameters
        self.model = self.make_model(env.parameters.grid_res, 8, parameters)
        self.target_model: Sequential = keras.models.clone_model(self.model)  # type: ignore
        self.replay_memory: "deque[MemoryEntry]" = deque(
            maxlen=self.parameters.replay_memory_max_size
        )
        self.train_count = 0
        self.epsilon = self.parameters.initial_epsilon

    def signature(self) -> str:
        return (
            f"dqns"
            f"-Gr{self.env.parameters.grid_res}"
            f"-Cr{self.env.parameters.cam_res}"
            f"-Rf{self.env.parameters.reward_forward_rate}"
            f"-Rr{self.env.parameters.reward_reverse_rate}"
            f"-Rs{self.env.parameters.reward_speeding_rate}"
            f"-Rd{self.env.parameters.reward_distance_rate}"
            f"-Of{self.env.parameters.out_of_track_fine}"
            f"-Cd{self.env.parameters.collision_distance}"
            f"-Ms{self.env.parameters.max_steps}"
            f"-Ro{self.env.parameters.max_random_rotation}"
            f"-D{self.parameters.discount}"
            f"-E{self.parameters.epsilon_decay}"
            f"-e{self.parameters.epsilon_min}"
            f"-M{self.parameters.replay_memory_max_size}"
            f"-m{self.parameters.replay_memory_min_size}"
            f"-B{self.parameters.minibatch_size}"
            f"-U{self.parameters.target_update_period}"
            f"-P{self.parameters.max_episodes}"
            f"-T{self.parameters.train_period}"
        )

    @staticmethod
    def control_to_action(turtle_name: str, control: int) -> Action:
        return Action(
            turtle_name,
            speed=0.4 if control >= 3 else 0.2,
            turn=[-0.25, 0.0, 0.25][control % 3],
        )

    @staticmethod
    def input_stack(last: TurtleCameraView, current: TurtleCameraView) -> NDArrayFloat:
        return np.stack(
            [
                current.penalties,
                current.distances_to_goal,
                current.speeds_along_azimuth,
                current.speeds_perpendicular_azimuth,
                last.penalties,
                last.distances_to_goal,
                last.speeds_along_azimuth,
                last.speeds_perpendicular_azimuth,
            ],
            axis=-1,
        )

    @staticmethod
    def decision(
        model: Sequential,
        last: TurtleCameraView,
        current: TurtleCameraView,
    ) -> NDArrayFloat:
        input = np.expand_dims(DQNSingle.input_stack(last, current), (-1, 0))
        return model(input).numpy().flatten()  # type: ignore

    @staticmethod
    def make_model(n: int, m: int, params: DQNParameters) -> Sequential:
        model: Any = Sequential()
        model.add(
            Conv3D(
                filters=2 * m, kernel_size=(2, 2, m), activation="relu", input_shape=(n, n, m, 1)
            )
        )
        model.add(Permute((1, 2, 4, 3)))
        model.add(Conv3D(filters=2 * m, kernel_size=(2, 2, 2 * m), activation="relu"))
        model.add(Permute((1, 2, 4, 3)))
        model.add(Conv3D(filters=2 * m, kernel_size=(2, 2, 2 * m), activation="relu"))
        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(params.control_dimension, activation="linear"))
        model.compile(
            loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]
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
            print(f"Episode {episode} (with {len(self.replay_memory)} moves in memory)")
            reward = self.train_episode(turtle_name, randomize_section)

            rewards.append(reward)
            print(f"\tmean reward up to and incl. this episode: {mean(rewards)}")

            # TODO: Studenci - okresowy zapis modelu
            if save_model and (episode + 1) % self.parameters.save_period == 0:
                self.save_model()

        self.save_model()

    def train_episode(self, turtle_name: str, randomize_section: bool = True) -> float:
        self.env.reset(turtle_names=[turtle_name], randomize_section=randomize_section)
        current_state = self.env.agents[turtle_name].camera_view
        last_state = current_state.copy()
        total_reward = 0.0

        while True:
            if self.random.random() > self.epsilon:
                print("\tsteering from model")
                control = int(np.argmax(self.decision(self.model, last_state, current_state)))
            else:
                print("\tsteering randomly")
                control = self.random.randint(0, self.parameters.control_dimension)

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
                print("\tminibatch training")
                self.train_minibatch()
                self.train_count += 1

                if self.train_count % self.parameters.target_update_period == 0:
                    print("\tupdating target model weights")
                    target_model.set_weights(self.model.get_weights())  # type: ignore

            if done:
                break

            last_state, current_state = current_state, new_state
            self.epsilon = max(
                self.parameters.epsilon_min,
                self.epsilon * self.parameters.epsilon_decay,
            )

        return total_reward

    def train_minibatch(self) -> None:
        moves = self.random.sample(self.replay_memory, self.parameters.minibatch_size)
        main_model_current_rewards = [
            self.decision(self.model, move.last_state, move.current_state) for move in moves
        ]
        target_model_next_rewards = [
            self.decision(self.target_model, move.current_state, move.new_state) for move in moves
        ]

        training_situations: List[NDArrayFloat] = []
        decision_situations: List[NDArrayFloat] = []

        for idx, move in enumerate(moves):
            new_reward = move.reward
            if not move.done:
                new_reward += self.parameters.discount * np.max(target_model_next_rewards[idx])  # type: ignore

            current_reward = main_model_current_rewards[idx].copy()
            current_reward[move.control] = new_reward

            training_situations.append(
                np.expand_dims(self.input_stack(move.last_state, move.current_state), axis=-1)
            )
            decision_situations.append(current_reward)

        self.model.fit(  # type: ignore
            x=np.stack(training_situations),
            y=np.stack(decision_situations),
            batch_size=self.parameters.training_batch_size,
            verbose=0,  # type: ignore
            shuffle=False,
        )

        # TODO: Studenci
        weights = np.copy(self.model.weights[0])  # type: ignore
        weights[:, :, 8:, :, :] = np.copy(self.model.weights[0][:, :, 8:, :, :])  # type: ignore
        self.model.weights[0] = tf.Variable(weights)  # type: ignore

    def save_model(self) -> None:
        self.model.save(MODELS_DIR / f"{self.signature()}.h5")  # type: ignore


if __name__ == "__main__":
    from .env_single import EnvSingle
    from .simulator import create_simulator

    with create_simulator() as simulator:
        env = EnvSingle(simulator)
        env.setup("routes.csv", agent_limit=1)

        turtle_name = next(iter(env.agents))

        dqn = DQNSingle(env)
        dqn.train(turtle_name, randomize_section=False)
