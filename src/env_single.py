from copy import copy
from dataclasses import dataclass
from math import cos, sin, sqrt
from random import uniform

from .env_base import Action, EnvBase, StepResult
from .simulator import Position


@dataclass
class EnvSingle(EnvBase):
    def single_step(self, action: Action, realtime: bool = False) -> StepResult:
        agent = self.agents[action.turtle_name]
        pose_before = copy(agent.pose)
        distance_to_goal_before = self.get_turtle_road_view(agent.name).distance_to_goal

        # TODO: Studenci - "przejechać 1/2 okresu, skręcić, przejechać pozostałą 1/2"
        if realtime:
            raise NotImplementedError("realtime mode")
        else:
            angle = agent.pose.angle + action.turn
            x = agent.pose.x + cos(angle) * action.speed * self.parameters.seconds_per_step
            y = agent.pose.y + sin(action.turn) * action.speed * self.parameters.seconds_per_step
            agent.pose = Position(x, y, angle)
            self.simulator.move_absolute(agent.name, agent.pose)

        road = self.get_turtle_road_view(agent.name)
        speed_x = (agent.pose.x - pose_before.x) / self.parameters.seconds_per_step
        speed_y = (agent.pose.y - pose_before.y) / self.parameters.seconds_per_step
        current_speed = sqrt(speed_x**2 + speed_y**2)
        suggested_speed = sqrt(road.speed_x**2 + road.speed_y**2)

        reward_speeding = min(
            0,
            self.parameters.reward_speeding_rate * (current_speed - suggested_speed),
        )
        if suggested_speed > 0.001:
            speed_ratio = (speed_x * road.speed_x + speed_y * road.speed_y) / suggested_speed
            if speed_ratio > 0:
                reward_direction = speed_ratio * self.parameters.reward_forward_rate
            else:
                reward_direction = speed_ratio * -self.parameters.reward_reverse_rate
        else:
            reward_direction = 0
        reward_distance = (
            road.distance_to_goal - distance_to_goal_before
        ) * self.parameters.reward_distance_rate
        reward_out_of_track = (
            self.parameters.out_of_track_fine
            if road.penalty == 1 and abs(road.speed_x) + abs(road.speed_y) < 0.01
            else 0
        )
        reward = (
            road.penalty * (reward_speeding + reward_direction)
            + reward_distance
            + reward_out_of_track
        )
        done = reward_out_of_track > 0 or self.step_sum > self.parameters.max_steps

        agent.camera_view = self.get_turtle_camera_view(agent.name, agent)
        return StepResult(agent.camera_view, reward, done)


if __name__ == "__main__":
    from .simulator import create_simulator

    with create_simulator() as simulator:
        env = EnvSingle(simulator)
        env.setup("routes.csv", agent_limit=1)
        env.reset()
        for _ in range(10):
            env.step(
                (Action(name, uniform(0.2, 1.0), uniform(-0.3, 0.3)) for name in env.agents),
                realtime=False,
            )