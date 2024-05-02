# Copyright (c) 2024 Mateusz Brzozowski, Bartłomiej Krawczyk, Mikołaj Kuranowski, Konrad Wojda
# SPDX-License-Identifier: MIT

from math import degrees
from pathlib import Path
from threading import Lock, Thread
from typing import Any, List, Optional

import pygame
from typing_extensions import Self

from .simple import SimpleSimulator
from .simulator import CameraCell, Position

DATA_DIR = Path(__file__).with_name("data")


class PygameSimulator(SimpleSimulator):
    def __init__(
        self,
        background_path: str = "roads.png",
        scale_m_to_px: float = 22,
    ) -> None:
        self.refresh_event_type = pygame.event.custom_type()
        self.pygame_thread: Optional[Thread] = None
        self.turtles_lock = Lock()
        super().__init__(background_path, scale_m_to_px)

    def __enter__(self) -> Self:
        assert self.pygame_thread is None
        self.pygame_thread = Thread(target=self.run_pygame_thread)
        self.pygame_thread.start()

        return super().__enter__()

    def __exit__(self, *_: Any) -> bool:
        if self.pygame_thread and self.pygame_thread.is_alive():
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            self.pygame_thread.join()
            self.pygame_thread = None
        return False

    def request_refresh(self) -> None:
        if self.pygame_thread and self.pygame_thread.is_alive():
            pygame.event.post(pygame.event.Event(self.refresh_event_type))
        else:
            raise RuntimeError("Request refreshed, but pygame is not running")

    def has_turtle(self, name: str) -> bool:
        with self.turtles_lock:
            return super().has_turtle(name)

    def kill_turtle(self, name: str) -> None:
        with self.turtles_lock:
            super().kill_turtle(name)
        self.request_refresh()

    def spawn_turtle(self, name: str, at: Position = Position()) -> None:
        with self.turtles_lock:
            super().spawn_turtle(name, at)
        self.request_refresh()

    def move_absolute(self, name: str, new_pos: Position) -> None:
        with self.turtles_lock:
            super().move_absolute(name, new_pos)
        self.request_refresh()

    def move_relative(self, name: str, distance: float, angle: float) -> None:
        with self.turtles_lock:
            super().move_relative(name, distance, angle)
        self.request_refresh()

    def get_position(self, name: str) -> Position:
        with self.turtles_lock:
            return super().get_position(name)

    def read_camera(
        self,
        name: str,
        frame_pixel_size: int,
        cell_side_count: int,
        goal: Position,
    ) -> List[List[CameraCell]]:
        with self.turtles_lock:
            return super().read_camera(name, frame_pixel_size, cell_side_count, goal)

    # NOTE: get_color_checker doesn't need to be overwritten. SimpleColorChecker will call
    #       simulator.get_position, which will acquire the turtles_lock.

    def run_pygame_thread(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode(
            (self.background.shape[1], self.background.shape[0]),
        )

        turtle_surface = pygame.image.load(DATA_DIR / "turtle.png").convert_alpha()
        assert turtle_surface.get_rect().width == turtle_surface.get_rect().height

        background = pygame.image.load(self.background_path).convert()
        screen.blit(background, (0, 0))
        pygame.display.update()

        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                break
            elif event.type == self.refresh_event_type:
                screen.blit(background, (0, 0))

                with self.turtles_lock:
                    for turtle_pos in self.turtles.values():
                        x, y = self.position_to_pixels(turtle_pos)
                        rotated_surface = pygame.transform.rotate(
                            turtle_surface,
                            degrees(turtle_pos.angle),
                        )
                        rotated_surface_rect = rotated_surface.get_rect()
                        x -= rotated_surface_rect.width / 2
                        y -= rotated_surface_rect.height / 2
                        screen.blit(rotated_surface, (x, y))

                pygame.display.update()

        pygame.quit()


if __name__ == "__main__":
    from math import pi
    from time import sleep

    with PygameSimulator() as simulator:
        simulator.spawn_turtle("a", Position(20, 20))
        while True:
            sleep(1)
            simulator.move_relative("a", 4, pi / 4)
