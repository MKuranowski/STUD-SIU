# pyright: basic

import logging
from pathlib import Path
from random import Random
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .env_base import Parameters
from .dqn_single import DQNParameters
from .play_single import PlaySingle

MODELS_DIR = Path("models")

NDArrayFloat = npt.NDArray[np.float_]

logger = logging.getLogger(__name__)


class ModelResult(NamedTuple):
    reward: float
    signature: str
    parameters: Parameters
    dqn_parameters: DQNParameters


def train(parameters: Parameters, dqn_parameters: DQNParameters) -> ModelResult:
    with create_simulator() as simulator:
        env = EnvSingle(
            simulator,
            parameters=parameters
        )
        env.setup("routes.csv", agent_limit=1)
        turtle_name = next(iter(env.agents))
        model = PlaySingle(
            env,
            parameters=dqn_parameters
        )
        model.train(turtle_name, randomize_section=True)
        model.random = Random(0)
        env.reset()
        reward = model.play_until_crash(max_laps=4)
        return ModelResult(
            reward=reward,
            signature=model.signature(),
            parameters=parameters,
            dqn_parameters=dqn_parameters,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    import coloredlogs

    from .env_single import EnvSingle
    from .simulator import create_simulator

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = arg_parser.parse_args()

    coloredlogs.install(level=logging.DEBUG if args.verbose else logging.INFO)

    seed = 42
    iterations = 4
    max_episodes = 10

    parameters_distributions = {
        'grid_res': range(4, 10),
        'cam_res': [200],
        'reward_forward_rate': [0.5, 1.0, 2.0, 4.0, 8.0],
        'reward_reverse_rate': [-10.0, -15.0, -20],
        'reward_speeding_rate': [-10.0, -15.0, -20.0],
        'reward_distance_rate': [2, 4, 8, 16],
        'out_of_track_fine': [-10.0, -15.0, -20.0],
        'max_steps': [10, 20, 40, 80],
        'goal_radius': [0.25, 0.5, 1.0, 1.5, 2.0],
    }
    dqn_parameters_distributions = {
        'discount': [0.9],
        'replay_memory_max_size': [20_000],
        'replay_memory_min_size': [4_000],
        'minibatch_size': [32],
        'training_batch_divisor': [4],
        'target_update_period': [20],
        'max_episodes': [max_episodes],
        'train_period': [4],
        'save_period': [max_episodes],
    }

    random = Random(seed)

    results: list[ModelResult] = []

    for iteration in range(iterations):
        parameters_from_distribution = {
            key: random.choice(values) for key, values in parameters_distributions.items()
        }
        parameters = Parameters(**parameters_from_distribution)
        dqn_parameters_from_distribution = {
            key: random.choice(values) for key, values in dqn_parameters_distributions.items()
        }
        dqn_parameters = DQNParameters(**dqn_parameters_from_distribution)
        logger.info('Iteration: %d', iteration)
        logger.info('DQN parameters: %s', dqn_parameters)
        logger.info('Environment parameters: %s', parameters)
        result = train(parameters, dqn_parameters)
        logger.info('Result: %s', result)
        results.append(result)

    for i, result in enumerate(sorted(results, key=lambda x: x.reward, reverse=True), start=1):
        logger.info('%d) %f %s', i, result.reward, result.signature)
