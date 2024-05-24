# Copyright (c) 2024 Mateusz Brzozowski, Bartłomiej Krawczyk, Mikołaj Kuranowski, Konrad Wojda
# SPDX-License-Identifier: MIT

# pyright: basic

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np

from .dqn_multi import DQNMulti
from .dqn_single import Episode
from .environment import Action, Parameters
from .play_single import extract_camera_res_from_filename, extract_grid_res_from_filename

logger = logging.getLogger(__name__)


class PlayMulti(DQNMulti):
    def play_until_crash(self, max_laps: Optional[int] = None) -> float:
        active_episodes = {
            name: Episode.for_agent(agent) for name, agent in self.env.agents.items()
        }
        crashed_episodes: Dict[str, Episode] = {}
        total_laps = {name: 0 for name in active_episodes}

        while active_episodes:
            # Set controls and actions
            for name, episode in active_episodes.items():
                episode.control = int(
                    np.argmax(
                        self.decision(
                            self.model,
                            episode.last_state,
                            episode.current_state,
                        )
                    )
                )
                episode.action = self.control_to_action(name, episode.control)

            # Execute the actions
            results_by_name = self.env.step(
                [cast(Action, i.action) for i in active_episodes.values()]
            )

            # Remember action results
            turtles_to_remove: List[str] = []
            for name, episode in active_episodes.items():
                episode.result = results_by_name[name]
                episode.advance()
                should_remove = False

                if episode.done and not self.env.goal_reached(name):
                    should_remove = True
                    logger.error(
                        "Turtle %s crashed after %d total laps - exiting",
                        name,
                        total_laps[name],
                    )
                elif episode.done:
                    agent = self.env.agents[name]
                    logger.debug("Turtle %s completed section %d", name, agent.section_id)
                    agent.section_id = (agent.section_id + 1) % len(agent.route)
                    agent.section = agent.route[agent.section_id]

                    if agent.section_id == 0:
                        total_laps[name] += 1
                        should_remove = max_laps is not None and total_laps[name] >= max_laps
                        logger.info(
                            "Turtle %s completed lap %d reward %f",
                            name,
                            total_laps[name],
                            episode.total_reward * (1 + total_laps[name]),
                        )

                if should_remove:
                    turtles_to_remove.append(name)

            # Remove crashed turtles
            for name in turtles_to_remove:
                self.env.simulator.kill_turtle(name)
                crashed_episodes[name] = active_episodes.pop(name)

        return sum(
            episode.total_reward * (1 + total_laps[name])
            for name, episode in crashed_episodes.items()
        )


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
        env.setup("routes.csv")
        env.reset()

        play = PlayMulti(env)
        play.load_model(args.model)
        indicator = play.play_until_crash()
        logger.info("Indicator: %.3f", indicator)
