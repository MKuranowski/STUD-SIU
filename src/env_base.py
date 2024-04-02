import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import atan2, cos, dist, inf, pi, sin
from random import uniform
from typing import Dict, Iterable, List, NamedTuple, Optional

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
    occupancy: NDArrayFloat = np.array(())

    def copy(self) -> "TurtleCameraView":
        return TurtleCameraView(
            self.speeds_x.copy(),
            self.speeds_y.copy(),
            self.penalties.copy(),
            self.distances_to_goal.copy(),
            self.speeds_along_azimuth.copy(),
            self.speeds_perpendicular_azimuth.copy(),
            self.occupancy.copy(),
        )


@dataclass
class TurtleAgent:
    name: str
    route: List[RouteSection]
    section_id: int
    section: RouteSection
    id_within_section: int
    color_api: ColorChecker
    pose: Position = Position()
    camera_view: TurtleCameraView = TurtleCameraView()


@dataclass(frozen=True)
class Parameters:
    grid_res: int = 5
    cam_res: int = 200
    seconds_per_step: float = 1.0
    reward_forward_rate: float = 0.5
    reward_reverse_rate: float = -10.0
    reward_speeding_rate: float = -10.0
    reward_distance_rate: float = 2.0
    out_of_track_fine: float = -10.0
    collision_distance: float = 1.5
    detect_collisions: bool = False
    max_steps: int = 20
    max_random_rotation: float = pi / 6


class Action(NamedTuple):
    turtle_name: str
    speed: float
    turn: float


class StepResult(NamedTuple):
    map: TurtleCameraView
    reward: float
    done: bool


@dataclass
class EnvBase(ABC):
    simulator: Simulator
    parameters: Parameters = Parameters()
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
        for route_id, sections in self.routes.items():
            for section_id, section in enumerate(sections):
                for agent_in_section_id in range(section.agents_no):
                    self.spawn_agent(route_id, section_id, agent_in_section_id)
                    agent_count += 1
                    if agent_count >= agent_limit:
                        return

    def spawn_agent(self, route_id: int, section_id: int, agent_in_section_id: int) -> None:
        route = self.routes[route_id]
        section = route[section_id]

        # NOTE: The rospy backend is stupid as *** and only accepts [A-Za-z/] characters
        name = "/".join(
            (int_to_ascii(route_id), int_to_ascii(section_id), int_to_ascii(agent_in_section_id))
        )
        if self.simulator.has_turtle(name):
            self.simulator.kill_turtle(name)
        self.simulator.spawn_turtle(name)

        agent = TurtleAgent(
            name=name,
            route=self.routes[route_id],
            section_id=section_id,
            section=section,
            id_within_section=agent_in_section_id,
            color_api=self.simulator.get_color_checker(name),
        )

        self.agents[name] = agent

    def reset(
        self,
        turtle_names: Optional[Iterable[str]] = None,
        randomize_section: bool = False,
    ) -> Dict[str, TurtleAgent]:
        self.step_sum = 0
        for turtle_name in turtle_names or self.agents.keys():
            self.reset_turtle(turtle_name, randomize_section)
        return self.agents

    def reset_turtle(self, turtle_name: str, randomize_section: bool = False) -> None:
        agent = self.agents[turtle_name]

        if randomize_section:
            # TODO: Studenci - losowanie obszaru proporcjonalnie do liczby planowanych żółwi w obszarze
            raise NotImplementedError

        while True:
            try:
                self.try_reset_turtle_within_section(turtle_name, agent)
                break
            except SpawnError:
                pass

    def try_reset_turtle_within_section(self, turtle_name: str, agent: TurtleAgent) -> None:
        x = uniform(agent.section.start_left, agent.section.start_right)
        y = uniform(agent.section.start_bottom, agent.section.start_top)
        theta = atan2(agent.section.goal_y - y, agent.section.goal_x - x)

        agent.pose = Position(x, y, theta)
        self.simulator.move_absolute(turtle_name, agent.pose)

        speed_x, speed_y, _, _, _, _ = self.get_turtle_road_view(turtle_name, agent)
        if self.parameters.detect_collisions:
            agent.camera_view = self.get_turtle_camera_view(turtle_name, agent)
            if (
                agent.camera_view.occupancy[
                    self.parameters.grid_res // 2, self.parameters.grid_res - 1
                ]
                == 0
            ):
                raise SpawnError("collision")

        if abs(speed_x) + abs(speed_y) <= 0.01:
            raise SpawnError("spawn at place with low suggested speed")

        theta += uniform(
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

        speed_x = (color.r - 200) / 50
        speed_y = (color.b - 200) / 50
        penalty = color.g / 255
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
        occupancy = self.base_camera_matrix()

        for i, row in enumerate(img):
            for j, cell in enumerate(row):
                speeds_x[i, j] = cell.r
                speeds_y[i, j] = cell.b
                penalties[i, j] = cell.g
                distances_to_goal[i, j] = cell.distance_to_goal
                occupancy[i, j] = cell.free

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
            occupancy,
        )

    def base_camera_matrix(self) -> NDArrayFloat:
        return np.zeros((self.parameters.grid_res, self.parameters.grid_res))

    def step(self, actions: Iterable[Action], realtime: bool = False) -> Dict[str, StepResult]:
        self.step_sum += 1
        return {action.turtle_name: self.single_step(action, realtime) for action in actions}

    @abstractmethod
    def single_step(self, action: Action, realtime: bool = False) -> StepResult:
        raise NotImplementedError


def int_to_ascii(n: int) -> str:
    if n < 0 or n > 25:
        raise ValueError("int_to_ascii only supports numbers between 0 and 25 (incl.)")
    return chr(0x41 + n)
