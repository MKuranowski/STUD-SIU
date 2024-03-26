import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import atan2, cos, dist, inf, pi, sin
from random import uniform
from time import sleep
from typing import Dict, Iterable, List, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
import rospy
from turtlesim.msg import Color, Pose
from turtlesim.srv import SetPenRequest
from TurtlesimSIU import ColorSensor, TurtlesimSIU

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


@dataclass
class TurtleAgent:
    name: str
    route: List[RouteSection]
    section_id: int
    section: RouteSection
    id_within_section: int
    color_api: ColorSensor
    pose: Pose = Pose()
    camera_view: TurtleCameraView = TurtleCameraView()


@dataclass
class Parameters:
    grid_res: int = 5
    cam_res: int = 200
    seconds_per_step: float = 1.0
    wait_after_move: Optional[float] = 0.005
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
    parameters: Parameters = Parameters()
    turtlesim_api: TurtlesimSIU = field(default_factory=TurtlesimSIU)
    meter_to_pixel_ratio: float = 22.0
    routes: Dict[int, List[RouteSection]] = field(default_factory=dict)
    agents: Dict[str, TurtleAgent] = field(default_factory=dict)
    step_sum: int = 0

    def setup(self, routes_filename: str, agent_limit: float = inf) -> None:
        rospy.init_node("siu_example", anonymous=False)  # type: ignore
        self.meter_to_pixel_ratio = self.turtlesim_api.pixelsToScale()
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
                    if agent_count > agent_limit:
                        return

    def spawn_agent(self, route_id: int, section_id: int, agent_in_section_id: int) -> None:
        route = self.routes[route_id]
        section = route[section_id]

        name = f"{route_id}/{section_id}/{agent_in_section_id}"
        if self.turtlesim_api.hasTurtle(name):
            self.turtlesim_api.killTurtle(name)
        self.turtlesim_api.spawnTurtle(name, Pose())
        self.turtlesim_api.setPen(name, SetPenRequest(off=1))

        agent = TurtleAgent(
            name=name,
            route=self.routes[route_id],
            section_id=section_id,
            section=section,
            id_within_section=agent_in_section_id,
            color_api=ColorSensor(name),
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
        agent.pose.x = uniform(agent.section.start_left, agent.section.start_right)
        agent.pose.y = uniform(agent.section.start_bottom, agent.section.start_top)
        agent.pose.theta = atan2(
            agent.section.goal_y - agent.pose.y,
            agent.section.goal_x - agent.pose.x,
        )
        self.turtlesim_api.setPose(turtle_name, agent.pose, "absolute")
        if self.parameters.wait_after_move:
            sleep(self.parameters.wait_after_move)

        speed_x, speed_y, _, _, _, _ = self.get_turtle_road_view(turtle_name, agent)
        agent.camera_view = self.get_turtle_camera_view(turtle_name, agent)
        if (
            self.parameters.detect_collisions
            and agent.camera_view.occupancy[
                self.parameters.grid_res // 2, self.parameters.grid_res - 1
            ]
            == 0
        ):
            raise SpawnError("collision")

        if abs(speed_x) + abs(speed_y) <= 0.01:
            raise SpawnError("spawn at place with low suggested speed")

        agent.pose.theta += uniform(
            -self.parameters.max_random_rotation,
            self.parameters.max_random_rotation,
        )
        self.turtlesim_api.setPose(turtle_name, agent.pose, "absolute")
        if self.parameters.wait_after_move:
            sleep(self.parameters.wait_after_move)

    def get_turtle_road_view(
        self,
        turtle_name: str,
        agent: Optional[TurtleAgent] = None,
    ) -> TurtleRoadView:
        agent = agent or self.agents[turtle_name]
        if self.parameters.wait_after_move:
            sleep(self.parameters.wait_after_move)

        color = agent.color_api.check() or Color(r=200, g=255, b=200)

        speed_x = (color.r - 200) / 50
        speed_y = (color.b - 200) / 50
        penalty = color.g / 255
        pose = self.turtlesim_api.getPose(turtle_name)

        distance_to_goal = dist((pose.x, pose.y), (agent.section.goal_x, agent.section.goal_y))
        speed_along_azimuth = speed_x * cos(pose.theta) + speed_y * sin(pose.theta)
        speed_perpendicular_azimuth = speed_y * cos(pose.theta) - speed_x * sin(pose.theta)

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
        pose = self.turtlesim_api.getPose(turtle_name)
        img = self.turtlesim_api.readCamera(
            name=turtle_name,
            frame_pixel_size=self.parameters.cam_res,
            cell_count=self.parameters.grid_res**2,
            x_offset=0,
            goal=Pose(agent.section.goal_x, agent.section.goal_y),
            show_matrix_cells_and_goal=False,
        )

        speeds_x = self.base_camera_matrix()
        speeds_y = self.base_camera_matrix()
        penalties = self.base_camera_matrix()
        distances_to_goal = self.base_camera_matrix()
        occupancy = self.base_camera_matrix()

        for i, row in enumerate(img.m_rows):
            for j, cell in enumerate(row.cells):
                speeds_x[i, j] = cell.red
                speeds_y[i, j] = cell.blue
                penalties[i, j] = cell.green
                distances_to_goal[i, j] = cell.distance
                occupancy[i, j] = cell.occupy

        speeds_along_azimuth = speeds_x * np.cos(pose.theta) + speeds_y * np.sin(pose.theta)
        speeds_perpendicular_azimuth = speeds_y * np.cos(pose.theta) - speeds_x * np.sin(
            pose.theta
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
