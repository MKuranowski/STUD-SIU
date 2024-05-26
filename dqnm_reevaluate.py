import logging
from argparse import ArgumentParser
from multiprocessing import Pool as ProcessPool
from pathlib import Path

import coloredlogs

from src.environment import Environment, Parameters
from src.play_multi import PlayMulti
from src.play_single import extract_camera_res_from_filename, extract_grid_res_from_filename
from src.simulator import create_simulator
from src.turtle_estimator import ModelResult, load_result_for_signature, save_result

logger = logging.getLogger("dqnm_reevaluate")


def reevaluate(model: Path, unknown_only: bool = False) -> None:
    _, hash, signature = model.stem.split("-", maxsplit=2)

    if unknown_only and load_result_for_signature(signature, multi=True) is not None:
        return

    print(f"{hash} - starting")
    parameters = Parameters(
        max_steps=None,
        grid_res=extract_grid_res_from_filename(model.name),
        cam_res=extract_camera_res_from_filename(model.name),
    )

    with create_simulator() as simulator:
        env = Environment(simulator, parameters)
        env.setup("routes.csv", agent_limit=14)
        play = PlayMulti(env)
        play.load_model(model)
        reward = play.evaluate(max_laps=4)

    print(f"{hash} - finished, reward: {reward:.3f}")
    save_result(ModelResult(reward, hash, signature), multi=True)


if __name__ == "__main__":
    coloredlogs.install(logging.INFO)

    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "-u",
        "--unknown-only",
        action="store_true",
        help="only reevaluate models not present in the CSV",
    )
    args = arg_parser.parse_args()

    with ProcessPool(maxtasksperchild=1) as pool:
        pool.starmap(
            reevaluate,
            ((model, args.unknown_only) for model in Path("models").glob("dqnm-*.h5")),
        )
