# pyright: basic

from pathlib import Path
import logging

import numpy as np

from .dqn_single import DQNSingle


logger = logging.getLogger(__name__)


class PlaySingle(DQNSingle):
    def play_until_crash(self, turtle_name: str = "") -> None:
        turtle_name = turtle_name or next(iter(self.env.agents))
        current_state = self.env.get_turtle_camera_view(turtle_name)
        last_state = current_state
        total_laps = 0

        while True:
            control = int(np.argmax(self.decision(self.model, last_state, current_state)))

            last_state = current_state
            current_state, _, done = self.env.step(
                [self.control_to_action(turtle_name, control)],
            )[turtle_name]

            if done and not self.env.goal_reached(turtle_name):
                logger.error("Turtle crashed after %d total laps - exiting", total_laps)
                break
            elif done:
                agent = self.env.agents[turtle_name]
                logger.debug("Section %d completed", agent.section_id)
                agent.section_id = (agent.section_id + 1) % len(agent.route)
                agent.section = agent.route[agent.section_id]

                if agent.section_id == 0:
                    total_laps += 1
                    logger.info("Lap %d completed", total_laps)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import coloredlogs

    from .env_single import EnvSingle
    from .simulator import create_simulator

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    arg_parser.add_argument("model", type=Path, help="path to model")
    args = arg_parser.parse_args()

    coloredlogs.install(level=logging.DEBUG if args.verbose else logging.INFO)

    with create_simulator() as simulator:
        env = EnvSingle(simulator)
        env.setup("routes.csv", agent_limit=1)
        env.reset()

        play = PlaySingle(env)
        play.load_model(args.model)
        play.play_until_crash()
