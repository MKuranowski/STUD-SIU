# Copyright (c) 2024 Mateusz Brzozowski, Bartłomiej Krawczyk, Mikołaj Kuranowski, Konrad Wojda
# SPDX-License-Identifier: MIT

# pyright: basic

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from .dqn_single import DQNSingle, Episode
from .environment import Parameters

logger = logging.getLogger(__name__)


class PlaySingle(DQNSingle):
    def play_until_crash(self, max_laps: Optional[int] = None) -> float:
        turtle_name = next(iter(self.env.agents))
        episode = Episode.for_agent(self.env.agents[turtle_name])
        total_laps = 0

        while not episode.done and (not max_laps or total_laps <= max_laps):
            episode.control = int(
                np.argmax(
                    self.decision(
                        self.model,
                        episode.last_state,
                        episode.current_state,
                    )
                )
            )
            episode.action = self.control_to_action(turtle_name, episode.control)
            episode.result = self.env.step([episode.action])[turtle_name]
            episode.advance()

            if episode.done and not self.env.goal_reached(turtle_name):
                logger.error("Turtle crashed after %d total laps - exiting", total_laps)
            elif episode.done:
                episode.done = False
                agent = self.env.agents[turtle_name]
                logger.debug("Section %d completed", agent.section_id)
                agent.section_id = (agent.section_id + 1) % len(agent.route)
                agent.section = agent.route[agent.section_id]

                if agent.section_id == 0:
                    total_laps += 1
                    logger.info(
                        "Lap %d completed with reward %f",
                        total_laps,
                        episode.total_reward * (1 + total_laps),
                    )

        return episode.total_reward * (1 + total_laps)


def extract_grid_res_from_filename(filename: str) -> int:
    match = re.search(r"Gr(\d+)", filename)
    if not match:
        raise ValueError("could not extract grid_res from filename")
    return int(match.group(1))


def extract_camera_res_from_filename(filename: str) -> int:
    match = re.search(r"Cr(\d+)", filename)
    if not match:
        raise ValueError("could not extract camera_res from filename")
    return int(match.group(1))


if __name__ == "__main__":
    from argparse import ArgumentParser

    import coloredlogs

    from .environment import Environment
    from .simulator import create_simulator

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    arg_parser.add_argument("model", type=Path, help="path to model")
    args = arg_parser.parse_args()

    parameters = Parameters(
        max_steps=None,
        grid_res=extract_grid_res_from_filename(args.model.name),
        cam_res=extract_camera_res_from_filename(args.model.name),
    )
    with create_simulator() as simulator:
        coloredlogs.install(level=logging.DEBUG if args.verbose else logging.INFO)
        env = Environment(simulator, parameters=parameters)
        env.setup("routes.csv", agent_limit=1)
        env.reset()

        play = PlaySingle(env)
        play.load_model(args.model)
        play.play_until_crash()
