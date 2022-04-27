from __future__ import annotations
import math
import arcade

from actorconfiguration import ActorConfiguration
from game_configuration import (
    FACTORY_BASE_HEALTH,
    START_RESOURCES,
    UNIT_BUILD_COST,
    UNITS_LIMIT,
)
from sprites.factory import Factory
from sprites.unit import Unit
import ai.utility_system.worldstate as worldstate

from ai.utility_system.planner import Planner


class Actor:
    def __init__(self, name: str, configuration: ActorConfiguration) -> None:
        self.name = name
        self.resources = START_RESOURCES
        self.configuration = configuration

        self.factory = Factory(
            f"{name}_{Factory.__name__}",
            self.configuration.building_sprite,
            *self.configuration.position,
        )

        self.units = arcade.SpriteList()
        self.planner = Planner()
        self.plan = None
        self.enemies_in_sight = []

    def replan(self, worldstate: worldstate.WorldState) -> None:
        self.plan = self.planner.find_action_with_highest_score(worldstate)

    def does_plan_exist(self) -> None:
        return self.plan != None

    def perform_plan(self, worldstate: worldstate.WorldState) -> None:
        if self.plan != None:
            self.plan.run_script(worldstate)
            self.plan = None

    def can_build_unit(self) -> bool:
        return self.resources >= UNIT_BUILD_COST and len(self.units) < UNITS_LIMIT

    def move_unit(self, unit_name: str, position_x: float, position_y: float) -> None:
        unit: Unit = next(filter(lambda unit: unit.name == unit_name, self.units), None)
        unit.move(position_x, position_y)

    def attack(self, unit_name: str, target: arcade.Sprite) -> None:
        unit: Unit = next(filter(lambda unit: unit.name == unit_name, self.units), None)
        unit.attack(target)

    def units_collisions_update(self, unit) -> None:
        collisions = arcade.check_for_collision_with_list(unit, self.units)

        for collision in collisions:
            x_diff = unit.center_x - collision.center_x
            y_diff = unit.center_y - collision.center_y
            angle = math.atan2(y_diff, x_diff)
            unit.center_x = unit.center_x + math.cos(angle)
            unit.center_y = unit.center_y + math.cos(angle)

    def get_status(self) -> str:
        return f"{self.name}: {self.resources} | {len(self.units)}/{UNITS_LIMIT} | {self.factory.health}/{FACTORY_BASE_HEALTH}"
