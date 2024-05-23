# pyright: basic

from typing import Dict, Mapping, Optional, Set

import keras
import numpy as np

from .dqn_single import DQNParameters, DQNSingle, MemoryEntry, NDArrayFloat
from .environment import Environment, StepResult, TurtleCameraView


class DQNMulti(DQNSingle):
    def __init__(
        self,
        env: Environment,
        parameters: DQNParameters = DQNParameters(),
        seed: int = 42,
        signature_prefix: str = "dqnm",
    ) -> None:
        super().__init__(env, parameters, seed, signature_prefix)
        self.last_states: Dict[str, TurtleCameraView] = {}
        self.current_states: Dict[str, TurtleCameraView] = {}

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

    def train_multi(self, save_model: bool = True, randomize_section: bool = True) -> None:
        self.replay_memory.clear()
        self.train_count = 0
        self.epsilon = self.parameters.initial_epsilon

        self.env.reset(randomize_section=randomize_section)
        episode = 0

        for name, agent in self.env.agents.items():
            self.last_states[name] = agent.camera_view
            self.current_states[name] = agent.camera_view

        while episode < self.parameters.max_episodes:
            done_turtles = self.train_until_first_done()

            self.env.reset(done_turtles, randomize_section)
            for _ in done_turtles:
                episode += 1
                self.on_episode_increment(episode, save_model)

    def train_until_first_done(self) -> Set[str]:
        while True:
            controls = {
                name: self.get_control(self.last_states[name], self.current_states[name])
                for name in self.env.agents
            }
            results = self.env.step(
                [self.control_to_action(name, control) for name, control in controls.items()],
            )
            self.train_after_actions(controls, results)
            if any(result.done for result in results.values()):
                return {name for name, result in results.items() if result.done}

    def train_after_actions(
        self,
        controls: Mapping[str, int],
        action_results: Mapping[str, StepResult],
    ) -> None:
        for name, result in action_results.items():
            self.replay_memory.append(
                MemoryEntry(
                    self.last_states[name],
                    self.current_states[name],
                    controls[name],
                    result.reward,
                    result.map,
                    result.done,
                )
            )

            if (
                len(self.replay_memory) >= self.parameters.replay_memory_min_size
                and self.env.step_sum % self.parameters.train_period == 0
            ):
                self.train_minibatch()
                self.train_count += 1

                if self.train_count % self.parameters.target_update_period == 0:
                    self.target_model.set_weights(self.model.get_weights())

            self.last_states[name] = self.current_states[name]
            self.current_states[name] = result.map
            self.epsilon = max(
                self.parameters.epsilon_min,
                self.epsilon * self.parameters.epsilon_decay,
            )
