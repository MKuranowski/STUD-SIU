# pyright: basic

import dataclasses
import logging
from itertools import count
from multiprocessing.pool import Pool
from operator import attrgetter
from pathlib import Path
from random import Random
from time import perf_counter
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .dqn_single import DQNParameters
from .env_base import Parameters
from .env_single import EnvSingle
from .play_single import PlaySingle
from .simulator import create_simulator

MODELS_DIR = Path("models")

NDArrayFloat = npt.NDArray[np.float_]

logger = logging.getLogger(__name__)


class ModelResult(NamedTuple):
    reward: float
    signature: str
    parameters: Parameters
    dqn_parameters: DQNParameters


def multithreaded_train(args: tuple[int, Parameters, DQNParameters]) -> ModelResult:
    logger.info("Starting iteration %d", args[0])

    start = perf_counter()
    result = train(args[1], args[2])
    elapsed = perf_counter() - start

    logger.info(
        "Iteration %d completed in %.2f s with reward %.3f",
        args[0],
        elapsed,
        result.reward,
    )
    return result


def train(parameters: Parameters, dqn_parameters: DQNParameters) -> ModelResult:
    with create_simulator() as simulator:
        env = EnvSingle(simulator, parameters=parameters)
        env.setup("routes.csv", agent_limit=1)
        turtle_name = next(iter(env.agents))
        model = PlaySingle(env, parameters=dqn_parameters)
        model.train(turtle_name, randomize_section=True)
        signature = model.signature()

        model.env.random = Random(0)
        restricted_parameters = dataclasses.asdict(parameters)
        restricted_parameters["max_steps"] = 4_000
        model.env.parameters = Parameters(**restricted_parameters)
        env.reset()
        reward = model.play_until_crash(max_laps=4)
        return ModelResult(
            reward=reward,
            signature=signature,
            parameters=parameters,
            dqn_parameters=dqn_parameters,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    import coloredlogs

    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=16,
        help="how many random parameters to test",
    )
    arg_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="seed for choosing parameters",
    )
    arg_parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for choosing parameters",
    )
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = arg_parser.parse_args()

    coloredlogs.install(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("src").setLevel(logging.INFO if args.verbose else logging.WARN)

    seed: int = args.seed
    iterations: int = args.iterations
    max_episodes = 4_000

    parameters_distributions = {
        "grid_res": [5, 7, 9],
        "cam_res": [200],
        "reward_forward_rate": [0.5, 1.0, 2.0, 4.0, 8.0],
        "reward_reverse_rate": [-10.0, -15.0, -20],
        "reward_speeding_rate": [-10.0, -15.0, -20.0],
        "reward_distance_rate": [2, 4, 8, 16],
        "out_of_track_fine": [-10.0, -15.0, -20.0],
        "max_steps": [10, 20, 40, 80],
        "goal_radius": [1.0],
    }
    dqn_parameters_distributions = {
        "discount": [0.8, 0.85, 0.9, 0.95],
        "replay_memory_max_size": [20_000],
        "replay_memory_min_size": [4_000],
        "minibatch_size": [32],
        "training_batch_divisor": [4],
        "target_update_period": [20],
        "max_episodes": [max_episodes],
        "train_period": [4],
        "save_period": [max_episodes],
    }

    random = Random(seed)

    parameters_from_distributions = [
        Parameters(
            **{key: random.choice(values) for key, values in parameters_distributions.items()}
        )
        for _ in range(iterations)
    ]

    dqn_parameters_from_distributions = [
        DQNParameters(
            **{key: random.choice(values) for key, values in dqn_parameters_distributions.items()}
        )
        for _ in range(iterations)
    ]

    with Pool(args.jobs, maxtasksperchild=1) as pool:
        results = pool.map(
            multithreaded_train,
            zip(count(), parameters_from_distributions, dqn_parameters_from_distributions),
        )

    results.sort(key=attrgetter("reward"), reverse=True)
    for i, result in enumerate(results, start=1):
        logger.info("(%02d) %f %s", i, result.reward, result.signature)
