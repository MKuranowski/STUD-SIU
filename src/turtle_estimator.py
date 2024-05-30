# Copyright (c) 2024 Mateusz Brzozowski, Bartłomiej Krawczyk, Mikołaj Kuranowski, Konrad Wojda
# SPDX-License-Identifier: MIT

# pyright: basic

import csv
import logging
from copy import copy
from hashlib import sha256
from itertools import count, repeat
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
from .play_multi import PlayMulti
from .play_single import PlaySingle
from .simulator import Simulator, create_simulator

MODELS_DIR = Path("models")
MODELS_CSV_SINGLE = Path("dqn_single_models.csv")
MODELS_CSV_MULTI = Path("dqn_multi_models.csv")

NDArrayFloat = npt.NDArray[np.float_]

logger = logging.getLogger(__name__)


class ModelResult(NamedTuple):
    reward: float
    prefix: str
    hash: str
    signature: str


def multithreaded_train(args: Tuple[int, Parameters, DQNParameters, bool, bool]) -> ModelResult:
    logger.info("Starting iteration %d", args[0])

    start = perf_counter()
    result = train(args[1], args[2], args[3], args[4])
    elapsed = perf_counter() - start

    logger.info(
        "Iteration %d completed in %.2f min with reward %.3f",
        args[0],
        elapsed / 60,
        result.reward,
    )
    return result


def train(
    parameters: Parameters,
    dqn_parameters: DQNParameters,
    multi: bool,
    partial: bool = False,
) -> ModelResult:
    signature = f"{parameters.signature()}_{dqn_parameters.signature()}"
    hash = sha256(signature.encode("ascii")).hexdigest()[:6]

    if result := load_result_for_signature(signature, multi):
        logger.error("%s was already evaluated", hash)
        return result

    with create_simulator() as simulator:
        env = Environment(simulator, parameters=copy(parameters))
        env.setup("routes.csv", agent_limit=14 if multi else 1)
        if multi:
            model = PlayMulti(
                env,
                parameters=dqn_parameters,
                signature_prefix="dqnc",
                episodes_without_collisions=2000 if partial else 0,
            )
        else:
            model = PlaySingle(env, parameters=dqn_parameters)
        model.train(save_model=False, randomize_section=not multi)
        path = model.save_path()

        model.env.parameters.max_steps = 4_000
        reward = evaluate(path, simulator, parameters, multi)

    result = ModelResult(reward, model.signature_prefix, hash, signature)
    save_result(result, multi)
    return result


def evaluate(path: Path, simulator: Simulator, parameters: Parameters, multi: bool) -> float:
    parameters = Parameters(
        max_steps=None,
        grid_res=parameters.grid_res,
        cam_res=parameters.cam_res,
    )
    env = Environment(simulator, parameters)
    env.setup("routes.csv", agent_limit=14 if multi else 1)
    env.reset()
    if multi:
        model = PlayMulti(env)
        model.load_model(path)
        return model.evaluate(max_laps=4)
    else:
        model = PlaySingle(env)
        model.load_model(path)
        return model.play_until_crash(max_laps=4)


def load_result_for_signature(signature: str, multi: bool = False) -> Optional[ModelResult]:
    csv_file = MODELS_CSV_MULTI if multi else MODELS_CSV_SINGLE
    lock_file = csv_file.with_suffix(".csv.lock")

    with FileLock(lock_file, timeout=10.0):
        return load_models_csv(csv_file).get(signature)


def save_result(result: ModelResult, multi: bool = False) -> None:
    csv_file = MODELS_CSV_MULTI if multi else MODELS_CSV_SINGLE
    lock_file = csv_file.with_suffix(".csv.lock")

    with FileLock(lock_file, timeout=10.0):
        results_by_signature = load_models_csv(csv_file)
        results_by_signature[result.signature] = result
        save_models_csv(csv_file, results_by_signature)


def load_models_csv(file: Path) -> Dict[str, ModelResult]:
    with file.open("r", encoding="ascii", newline="") as f:
        return {
            i["signature"]: ModelResult(float(i["reward"]), i["prefix"], i["hash"], i["signature"])
            for i in csv.DictReader(f)
        }


def save_models_csv(file: Path, results_by_signature: Dict[str, ModelResult]) -> None:
    with file.open("w", encoding="ascii", newline="") as f:
        w = csv.writer(f)
        w.writerow(("reward", "prefix", "hash", "signature"))
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
    arg_parser.add_argument("-m", "--multi", action="store_true", help="multi-agent estimation")
    arg_parser.add_argument(
        "-p",
        "--partial",
        action="store_true",
        help="run first half of episodes without collisions (only applicable with --multi)",
    )
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = arg_parser.parse_args()

    coloredlogs.install(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("src").setLevel(logging.INFO if args.verbose else logging.WARN)

    seed: int = args.seed
    iterations: int = args.iterations
    max_episodes = 4_000

    parameters_distributions = {
        "grid_res": [7, 9],
        "cam_res": [200, 250, 300],
        "reward_forward_rate": [0.1, 0.5, 1.0],
        "reward_reverse_rate": [-10.0, -15.0, -20.0],
        "reward_speeding_rate": [-10.0, -15.0],
        "reward_distance_rate": [4.0, 8.0, 16.0, 24.0, 32.0],
        "out_of_track_fine": [-10.0, -20.0, 30.0],
        "max_steps": [40, 60],
    }
    dqn_parameters_distributions = {
        "discount": [0.8, 0.85, 0.9],
        "save_period": [250],
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
            zip(
                count(),
                parameters_from_distributions,
                dqn_parameters_from_distributions,
                repeat(args.multi),
                repeat(args.partial),
            ),
        )
