from pathlib import Path
from threading import Thread
from typing import Any, Optional

import pygame
from typing_extensions import Self

from .simple import SimpleSimulator
from .simulator import Position

DATA_DIR = Path(__file__).with_name("data")


class PygameSimulator(SimpleSimulator):
    def __init__(
        self,
        background_path: str = "roads.png",
        scale_m_to_px: float = 22,
    ) -> None:
        self.refresh_event_type = pygame.event.custom_type()
        self.pygame_thread: Optional[Thread] = None
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
            raise RuntimeError("Request refreshed, but pygame is not")

    def kill_turtle(self, name: str) -> None:
        self.request_refresh()
        return super().kill_turtle(name)

    def spawn_turtle(self, name: str, at: Position = Position()) -> None:
        self.request_refresh()
        return super().spawn_turtle(name, at)

    def move_absolute(self, name: str, new_pos: Position) -> None:
        self.request_refresh()
        return super().move_absolute(name, new_pos)

    def move_relative(self, name: str, distance: float, angle: float) -> None:
        self.request_refresh()
        return super().move_relative(name, distance, angle)

    def run_pygame_thread(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode(
            (simulator.background.shape[1], simulator.background.shape[0]),
        )

        turtle_surface = pygame.image.load(DATA_DIR / "turtle.png").convert()
        assert turtle_surface.get_rect().width == turtle_surface.get_rect().height

        background = pygame.image.load(self.background_path).convert()
        screen.blit(background, (0, 0))
        pygame.display.update()

        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                print("Quitting")
                break
            elif event.type == simulator.refresh_event_type:
                screen.blit(background, (0, 0))
                for turtle in self.turtles.values():
                    x, y = self.position_to_pixels(turtle)
                    screen.blit(turtle_surface, (x, y))
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
