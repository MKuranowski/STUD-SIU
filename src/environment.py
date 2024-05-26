# Copyright (c) 2024 Mateusz Brzozowski, Bartłomiej Krawczyk, Mikołaj Kuranowski, Konrad Wojda
# SPDX-License-Identifier: MIT

import csv
import random
from dataclasses import dataclass, field
from itertools import combinations
from math import atan2, cos, dist, inf, pi, sin, sqrt
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt

from .simulator import ColorChecker, Position, Simulator

NDArrayFloat = npt.NDArray[np.float_]


class SpawnError(ValueError):
    pass


@dataclass
class RouteSection:
    route_id: int
    agents_no: int
    start_left: float
    start_right: float
    start_bottom: float
    start_top: float
    goal_x: float
    goal_y: float


class TurtleRoadView(NamedTuple):
    speed_x: float = 0.0
    speed_y: float = 0.0
    penalty: float = 0.0
    distance_to_goal: float = 0.0
    speed_along_azimuth: float = 0.0
    speed_perpendicular_azimuth: float = 0.0


class TurtleCameraView(NamedTuple):
    speeds_x: NDArrayFloat = np.array(())
    speeds_y: NDArrayFloat = np.array(())
    penalties: NDArrayFloat = np.array(())
    distances_to_goal: NDArrayFloat = np.array(())
    speeds_along_azimuth: NDArrayFloat = np.array(())
    speeds_perpendicular_azimuth: NDArrayFloat = np.array(())
    free_of_other_agents: NDArrayFloat = np.array(())

    def copy(self) -> "TurtleCameraView":
        return TurtleCameraView(
            self.speeds_x.copy(),
            self.speeds_y.copy(),
            self.penalties.copy(),
            self.distances_to_goal.copy(),
            self.speeds_along_azimuth.copy(),
            self.speeds_perpendicular_azimuth.copy(),
            self.free_of_other_agents.copy(),
        )

    def is_collision_likely(self) -> bool:
        return not self.free_of_other_agents[
            self.free_of_other_agents.shape[0] // 2,
            -1,
        ]


@dataclass
class TurtleAgent:
    name: str
    route: List[RouteSection]
    section_id: int
    section: RouteSection
    color_api: ColorChecker
    pose: Position = Position()
    camera_view: TurtleCameraView = TurtleCameraView()
    step_sum: int = 0


@dataclass
class Parameters:
    grid_res: int = 5
    """Resolution of the turtle's camera view.
    A cam_res by cam_res windows is compressed into a grid_res by grid_res cell matrix.

    Tweakable, default 5.
    """

    cam_res: int = 200
    """Resolution of the turtle's camera, in pixels.
    Turtles will be able to see an image cam_res by cam_res pixels in front of them.

    Tweakable, default 200.
    """

    seconds_per_step: float = 1.0
    """Time-wise steering discretization - how much time should pass
    between steps, in seconds.

    Not tweakable.
    """

    reward_forward_rate: float = 0.5
    """Reward rate for moving alongside the suggested axis.

    Tweakable, at least 0.5.
    """

    reward_reverse_rate: float = -10.0
    """Reward rate for moving opposite to the suggested axis.

    Tweakable, at most -10.
    """

    reward_speeding_rate: float = -10.0
    """Reward rate for moving with a higher-than-suggested speed.

    Tweakable, at most -10.
    """

    reward_distance_rate: float = 2.0
    """Reward rate for moving towards the goal.

    Tweakable, at least 2.
    """

    out_of_track_fine: float = -10.0
    """Total reward for falling outside of the track.

    Tweakable, at most -10.
    """

    collision_distance: float = 1.5
    """Distance between two turtles which cause a collision to be detected, in meters.

    Not tweakable.
    """

    detect_collisions: bool = True
    """Enable collision checking."""

    max_steps: Optional[int] = 200
    """Max steps for a turtle to reach its goal.

    Tweakable, default 20.
    """

    max_random_rotation: float = pi / 6
    """Maximum deviation for turtle's initial heading, radians.
    Turtles spawn heading dead-on towards the goal, with a uniformly-chosen deviation
    in the range [-max_random_rotation, max_random_rotation].

    Not tweakable.
    """

    goal_radius: float = 2.0
    """How close does an agent has to be to its goal to assume
    that the goal has been reached, in meters?

    Non-standard, tweakable, default 2.
    """

    def signature(self) -> str:
        return (
            f"Gr{self.grid_res}"
            f"_Cr{self.cam_res}"
            f"_Sw{self.reward_forward_rate}"
            f"_Sv{self.reward_reverse_rate}"
            f"_Sf{self.reward_speeding_rate}"
            f"_Dr{self.reward_distance_rate}"
            f"_Oo{self.out_of_track_fine}"
            f"_Ms{self.max_steps}"
            f"_Pb{pi / self.max_random_rotation:.0f}"
        )


class Action(NamedTuple):
    turtle_name: str
    speed: float
    turn: float


class StepResult(NamedTuple):
    map: TurtleCameraView
    reward: float
    done: bool


@dataclass
class AgentDataBeforeMove:
    pose: Position
    distance_to_goal: float


@dataclass
class RewardCalculator:
    agent: TurtleAgent
    road: TurtleRoadView
    before: AgentDataBeforeMove
    collided: bool
    parameters: Parameters

    speed_x: float = field(init=False)
    speed_y: float = field(init=False)
    current_speed: float = field(init=False)
    suggested_speed: float = field(init=False)

    def __post_init__(self) -> None:
        self.speed_x = (self.agent.pose.x - self.before.pose.x) / self.parameters.seconds_per_step
        self.speed_y = (self.agent.pose.y - self.before.pose.y) / self.parameters.seconds_per_step
        self.current_speed = sqrt(self.speed_x**2 + self.speed_y**2)
        self.suggested_speed = sqrt(self.road.speed_x**2 + self.road.speed_y**2)

    def calculate(self) -> Tuple[float, bool]:
        total_reward = (
            self.speeding_reward()
            + self.direction_reward()
            + self.distance_reward()
            + self.out_of_track_reward()
            + self.collision_reward()
        )
        done = self.is_done()
        return total_reward, done

    def speeding_reward(self) -> float:
        return min(
            0,
            self.parameters.reward_speeding_rate * (self.current_speed - self.suggested_speed),
        )

    def direction_reward(self) -> float:
        if self.suggested_speed > 0.001:
            speed_ratio = (
                self.speed_x * self.road.speed_x + self.speed_y * self.road.speed_y
            ) / self.suggested_speed
            if speed_ratio > 0:
                return self.parameters.reward_forward_rate * speed_ratio
            else:
                return self.parameters.reward_reverse_rate * -speed_ratio
        else:
            return 0

    def distance_reward(self) -> float:
        return self.parameters.reward_distance_rate * (
            self.before.distance_to_goal - self.road.distance_to_goal
        )

    def out_of_track_reward(self) -> float:
        return self.parameters.out_of_track_fine if self.is_out_of_track() else 0.0

    def collision_reward(self) -> float:
        return self.parameters.out_of_track_fine if self.collided else 0.0

    def is_out_of_track(self) -> bool:
        return self.road.penalty > 0.95 and abs(self.road.speed_x) + abs(self.road.speed_y) < 0.01

    def is_out_of_steps(self) -> bool:
        return (
            self.parameters.max_steps is not None
            and self.agent.step_sum > self.parameters.max_steps
        )

    def is_done(self) -> bool:
        return (
            self.collided
            or self.road.distance_to_goal <= self.parameters.goal_radius
            or self.is_out_of_steps()
            or self.is_out_of_track()
        )


@dataclass
class Environment:
    simulator: Simulator
    parameters: Parameters = field(default_factory=Parameters)
    routes: Dict[int, List[RouteSection]] = field(default_factory=dict)
    agents: Dict[str, TurtleAgent] = field(default_factory=dict)
    step_sum: int = 0

    def setup(self, routes_filename: str, agent_limit: float = inf) -> None:
        self.load_routes_from_file(routes_filename)
        self.create_agents(agent_limit)

    def load_routes_from_file(self, filename: str) -> None:
        self.routes.clear()
        with open(filename, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                id = int(row["id"])
                self.routes.setdefault(id, []).append(
                    RouteSection(
                        route_id=id,
                        agents_no=int(row["agents_no"]),
                        start_left=float(row["start_left"]),
                        start_right=float(row["start_right"]),
                        start_bottom=float(row["start_bottom"]),
                        start_top=float(row["start_top"]),
                        goal_x=float(row["goal_x"]),
                        goal_y=float(row["goal_y"]),
                    ),
                )

    def create_agents(self, agent_limit: float = inf) -> None:
        agent_count = 0
        spots = [
            (route_id, section_id)
            for route_id, sections in self.routes.items()
            for section_id, section in enumerate(sections)
            for _ in range(section.agents_no)
        ]
        random.shuffle(spots)
        for route_id, section_id in spots:
            self.spawn_agent(route_id, section_id, agent_id=agent_count)
            agent_count += 1
            if agent_count >= agent_limit:
                return

    def spawn_agent(self, route_id: int, section_id: int, agent_id: int) -> None:
        route = self.routes[route_id]
        section = route[section_id]

        # NOTE: The rospy backend is stupid as *** and only accepts [A-Za-z/] characters in names
        name = int_to_ascii(agent_id)
        if self.simulator.has_turtle(name):
            self.simulator.kill_turtle(name)
        self.simulator.spawn_turtle(name)

        agent = TurtleAgent(
            name=name,
            route=self.routes[route_id],
            section_id=section_id,
            section=section,
            color_api=self.simulator.get_color_checker(name),
        )

        self.agents[name] = agent

    def reset(
        self,
        turtle_names: Optional[Iterable[str]] = None,
        randomize_section: bool = False,
    ) -> Dict[str, TurtleAgent]:
        if turtle_names is None:
            self.step_sum = 0

        for turtle_name in turtle_names or self.agents.keys():
            self.reset_turtle(turtle_name, randomize_section)
        return self.agents

    def reset_turtle(self, turtle_name: str, randomize_section: bool = False) -> None:
        agent = self.agents[turtle_name]
        agent.step_sum = 0
        tries = 0

        while True:
            if randomize_section:
                agent.section_id = random.randint(0, len(agent.route) - 1)
                agent.section = agent.route[agent.section_id]
            try:
                self.try_reset_turtle_within_section(turtle_name, agent)
                break
            except SpawnError as e:
                tries += 1
                if tries >= 1_000:
                    raise SpawnError(
                        f"failed to spawn turtle {turtle_name} after {tries} tries",
                    ) from e

        agent.camera_view = self.get_turtle_camera_view(turtle_name, agent)

    def try_reset_turtle_within_section(self, turtle_name: str, agent: TurtleAgent) -> None:
        x = random.uniform(agent.section.start_left, agent.section.start_right)
        y = random.uniform(agent.section.start_bottom, agent.section.start_top)
        theta = atan2(agent.section.goal_y - y, agent.section.goal_x - x)

        agent.pose = Position(x, y, theta)
        self.simulator.move_absolute(turtle_name, agent.pose)

        speed_x, speed_y, _, _, _, _ = self.get_turtle_road_view(turtle_name, agent)
        if (
            self.parameters.detect_collisions
            and turtle_name in self.find_collided_agents()
        ):
            raise SpawnError("collision")

        if abs(speed_x) + abs(speed_y) <= 0.01:
            raise SpawnError("spawn at place with low suggested speed")

        theta += random.uniform(
            -self.parameters.max_random_rotation,
            self.parameters.max_random_rotation,
        )
        agent.pose = Position(x, y, theta)
        self.simulator.move_absolute(turtle_name, agent.pose)

    def get_turtle_road_view(
        self,
        turtle_name: str,
        agent: Optional[TurtleAgent] = None,
    ) -> TurtleRoadView:
        agent = agent or self.agents[turtle_name]
        # There was a sleep here, not sure if it necessary

        color = agent.color_api.check()

        speed_x = color.r
        speed_y = color.b
        penalty = color.g
        pose = self.simulator.get_position(turtle_name)

        distance_to_goal = dist((pose.x, pose.y), (agent.section.goal_x, agent.section.goal_y))
        speed_along_azimuth = speed_x * cos(pose.angle) + speed_y * sin(pose.angle)
        speed_perpendicular_azimuth = speed_y * cos(pose.angle) - speed_x * sin(pose.angle)

        return TurtleRoadView(
            speed_x,
            speed_y,
            penalty,
            distance_to_goal,
            speed_along_azimuth,
            speed_perpendicular_azimuth,
        )

    def get_turtle_camera_view(
        self,
        turtle_name: str,
        agent: Optional[TurtleAgent] = None,
    ) -> TurtleCameraView:
        agent = agent or self.agents[turtle_name]
        pose = self.simulator.get_position(turtle_name)
        img = self.simulator.read_camera(
            name=turtle_name,
            frame_pixel_size=self.parameters.cam_res,
            cell_side_count=self.parameters.grid_res,
            goal=Position(x=agent.section.goal_x, y=agent.section.goal_y),
        )

        speeds_x = self.base_camera_matrix()
        speeds_y = self.base_camera_matrix()
        penalties = self.base_camera_matrix()
        distances_to_goal = self.base_camera_matrix()
        free_of_other_agents = self.base_camera_matrix()

        for i, row in enumerate(img):
            for j, cell in enumerate(row):
                speeds_x[i, j] = cell.r
                speeds_y[i, j] = cell.b
                penalties[i, j] = cell.g
                distances_to_goal[i, j] = cell.distance_to_goal
                free_of_other_agents[i, j] = cell.free

        speeds_along_azimuth = speeds_x * np.cos(pose.angle) + speeds_y * np.sin(pose.angle)
        speeds_perpendicular_azimuth = speeds_y * np.cos(pose.angle) - speeds_x * np.sin(
            pose.angle
        )

        return TurtleCameraView(
            speeds_x,
            speeds_y,
            penalties,
            distances_to_goal,
            speeds_along_azimuth + 1,
            speeds_perpendicular_azimuth + 1,
            free_of_other_agents,
        )

    def base_camera_matrix(self) -> NDArrayFloat:
        return np.zeros((self.parameters.grid_res, self.parameters.grid_res))

    def goal_reached(self, turtle_name: str, agent: Optional[TurtleAgent] = None) -> bool:
        agent = agent or self.agents[turtle_name]
        distance_to_goal = dist(
            (agent.pose.x, agent.pose.y),
            (agent.section.goal_x, agent.section.goal_y),
        )
        return distance_to_goal <= self.parameters.goal_radius

    def step(self, actions: Sequence[Action], realtime: bool = False) -> Dict[str, StepResult]:
        self.step_sum += 1

        before = self.move_agents(actions, realtime)
        collided_agents: Set[str] = (
            self.find_collided_agents() if self.parameters.detect_collisions else set()
        )
        return {
            action.turtle_name: self.calculate_step_result(
                action.turtle_name,
                before[action.turtle_name],
                collided=action.turtle_name in collided_agents,
            )
            for action in actions
        }

    def move_agents(
        self,
        actions: Iterable[Action],
        realtime: bool = False,
    ) -> Dict[str, AgentDataBeforeMove]:
        return {action.turtle_name: self.move_agent(action, realtime) for action in actions}

    def move_agent(
        self,
        action: Action,
        realtime: bool = False,
        agent: Optional[TurtleAgent] = None,
    ) -> AgentDataBeforeMove:
        agent = agent or self.agents[action.turtle_name]
        agent.step_sum += 1
        before = AgentDataBeforeMove(
            agent.pose,
            self.get_turtle_road_view(agent.name).distance_to_goal,
        )

        # TODO: Studenci - "przejechać 1/2 okresu, skręcić, przejechać pozostałą 1/2"
        if realtime:
            self.simulator.move_relative(agent.name, action.speed * 0.5, 0.0)
            self.simulator.move_relative(agent.name, 0.0, pi - 2 * action.turn)
            self.simulator.move_relative(agent.name, action.speed * 0.5, 0.0)
            agent.pose = self.simulator.get_position(agent.name)
        else:
            angle = agent.pose.angle + action.turn
            x = agent.pose.x + cos(angle) * action.speed * self.parameters.seconds_per_step
            y = agent.pose.y + sin(angle) * action.speed * self.parameters.seconds_per_step
            agent.pose = Position(x, y, angle)
            self.simulator.move_absolute(agent.name, agent.pose)

        return before

    def find_collided_agents(self) -> Set[str]:
        collided_agents: Set[str] = set()
        for names in combinations(self.agents, 2):
            agents = (self.agents[name] for name in names)
            coordinates = ((agent.pose.x, agent.pose.y) for agent in agents)
            distance = dist(*coordinates)
            if distance < self.parameters.collision_distance:
                collided_agents.update(names)
        return collided_agents

    def calculate_step_result(
        self,
        agent_name: str,
        before: AgentDataBeforeMove,
        collided: bool = False,
        agent: Optional[TurtleAgent] = None,
    ) -> StepResult:
        agent = agent or self.agents[agent_name]
        road = self.get_turtle_road_view(agent.name)

        reward, done = RewardCalculator(agent, road, before, collided, self.parameters).calculate()

        agent.camera_view = self.get_turtle_camera_view(agent.name, agent)
        # if collided:
        #     assert agent.camera_view.is_collision_likely()

        return StepResult(agent.camera_view, reward, done)


def int_to_ascii(n: int) -> str:
    if n < 0 or n > 25:
        raise ValueError("int_to_ascii only supports numbers between 0 and 25 (incl.)")
    return chr(0x41 + n)
