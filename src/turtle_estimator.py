# Copyright (c) 2024 Mateusz Brzozowski, Bartłomiej Krawczyk, Mikołaj Kuranowski, Konrad Wojda
# SPDX-License-Identifier: MIT

# pyright: basic

import csv
import logging
from copy import copy
from hashlib import sha256
from itertools import count
from multiprocessing.pool import Pool
from operator import attrgetter
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
from filelock import FileLock

from .dqn_single import DQNParameters
from .environment import Environment, Parameters
from .play_single import PlaySingle
from .simulator import create_simulator

MODELS_DIR = Path("models")
MODELS_CSV = Path("dqn_single_models.csv")
MODELS_CSV_LOCK = MODELS_CSV.with_suffix(".csv.lock")

NDArrayFloat = npt.NDArray[np.float_]

logger = logging.getLogger(__name__)


class ModelResult(NamedTuple):
    reward: float
    hash: str
    signature: str


def multithreaded_train(args: Tuple[int, Parameters, DQNParameters]) -> ModelResult:
    logger.info("Starting iteration %d", args[0])

    start = perf_counter()
    result = train(args[1], args[2])
    elapsed = perf_counter() - start

    logger.info(
        "Iteration %d completed in %.2f min with reward %.3f",
        args[0],
        elapsed / 60,
        result.reward,
    )
    return result


def train(parameters: Parameters, dqn_parameters: DQNParameters) -> ModelResult:
    signature = f"{parameters.signature()}_{dqn_parameters.signature()}"
    hash = sha256(signature.encode("ascii")).hexdigest()[:6]

    if result := load_result_for_signature(signature):
        logger.error("%s was already evaluated", hash)
        return result

    with create_simulator() as simulator:
        env = Environment(simulator, parameters=copy(parameters))
        env.setup("routes.csv", agent_limit=1)
        model = PlaySingle(env, parameters=dqn_parameters)
        model.train(save_model=False, randomize_section=True)

        model.env.parameters.max_steps = 4_000
        env.reset()
        reward = model.play_until_crash(max_laps=4)

    result = ModelResult(reward, hash, signature)
    save_result(result)
    return result


def load_result_for_signature(signature: str) -> Optional[ModelResult]:
    with FileLock(MODELS_CSV_LOCK):
        return load_models_csv().get(signature)


def save_result(result: ModelResult):
    with FileLock(MODELS_CSV_LOCK):
        results_by_signature = load_models_csv()
        results_by_signature[result.signature] = result
        save_models_csv(results_by_signature)


def load_models_csv() -> Dict[str, ModelResult]:
    with MODELS_CSV.open("r", encoding="ascii", newline="") as f:
        return {
            i["signature"]: ModelResult(float(i["reward"]), i["hash"], i["signature"])
            for i in csv.DictReader(f)
        }


def save_models_csv(results_by_signature: Dict[str, ModelResult]):
    with MODELS_CSV.open("w", encoding="ascii", newline="") as f:
        w = csv.writer(f)
        w.writerow(("reward", "hash", "signature"))
        w.writerows(sorted(results_by_signature.values(), key=attrgetter("reward"), reverse=True))


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
        help="how may processes can run simultaneously",
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
        "cam_res": [50, 100, 200, 300],
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
        "train_period": [4],
    }

    random = Random(seed)

    parameters_from_distributions = (
        Parameters(
            **{key: random.choice(values) for key, values in parameters_distributions.items()}
        )
        for _ in range(iterations)
    )

    dqn_parameters_from_distributions = (
        DQNParameters(
            **{key: random.choice(values) for key, values in dqn_parameters_distributions.items()}
        )
        for _ in range(iterations)
    )

    with Pool(args.jobs, maxtasksperchild=1) as pool:
        results = pool.map(
            multithreaded_train,
            zip(count(), parameters_from_distributions, dqn_parameters_from_distributions),
        )
