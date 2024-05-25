import logging
from multiprocessing import Pool as ProcessPool
from pathlib import Path

import coloredlogs

from src.environment import Environment, Parameters
from src.play_multi import PlayMulti
from src.play_single import extract_camera_res_from_filename, extract_grid_res_from_filename
from src.simulator import create_simulator
from src.turtle_estimator import ModelResult, save_result

logger = logging.getLogger("dqnm_reevaluate")


def reevaluate(model: Path) -> None:
    _, hash, signature = model.name.split("-", maxsplit=2)
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
        reward = play.evaluate()

    print(f"{hash} - finished, reward: {reward:.3f}")
    save_result(ModelResult(reward, hash, signature), multi=True)


if __name__ == "__main__":
    coloredlogs.install(logging.INFO)

    with ProcessPool(maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(reevaluate, Path("models").glob("dqnm-*.h5")):
            pass
