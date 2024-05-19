import csv
import io
import sys
from typing import Any, Dict, cast

from src.simulator.simple import Position, SimpleSimulator

assert isinstance(sys.stdin, io.TextIOWrapper)
assert isinstance(sys.stdout, io.TextIOWrapper)
sys.stdin.reconfigure(encoding="utf-8", newline="")
sys.stdout.reconfigure(encoding="utf-8", newline="")

simulator = SimpleSimulator()

input = csv.DictReader(sys.stdin)
assert input.fieldnames is not None
output = csv.DictWriter(sys.stdout, input.fieldnames, delimiter=";")
output.writeheader()

for row in input:
    row = cast(Dict[str, Any], row)

    # Convert numeric fields to float
    for field, value in row.items():
        if field.startswith(("start", "goal")):
            row[field] = float(value)

    # Convert points from pixels to positions
    row["start_left"], row["start_bottom"] = simulator.pixels_to_position(
        row["start_left"],
        row["start_bottom"],
    )
    row["start_right"], row["start_top"] = simulator.pixels_to_position(
        row["start_right"],
        row["start_top"],
    )
    row["goal_x"], row["goal_y"] = simulator.pixels_to_position(
        row["goal_x"],
        row["goal_y"],
    )

    output.writerow(row)
