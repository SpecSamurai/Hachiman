from __future__ import annotations
import arcade

from actorconfiguration import ActorConfiguration
from sprites.factory import Factory
from sprites.unit import Unit
import ai.utility_system.worldstate as worldstate

from ai.utility_system.planner import Planner


class Actor:
    def __init__(self, name: str, configuration: ActorConfiguration) -> None:
        self.name = name
        self.resources = ActorConfiguration.START_RESOURCES
        self.configuration = configuration

        self.factory = Factory(
            name + "main_base",
            self.configuration.building_sprite,
            *self.configuration.position,
        )

        self.units = arcade.SpriteList()
        self.planner = Planner()
        self.plan = None
        self.enemies_in_sight = []

    def replan(self, worldstate: worldstate.WorldState) -> None:
        self.plan = Planner().find_action_with_highest_score(worldstate)

    def does_plan_exist(self) -> None:
        return self.plan != None

    def perform_plan(self, worldstate: worldstate.WorldState) -> None:
        if self.plan != None:
            self.plan.run_script(worldstate)
            self.plan = None

    def can_build_unit(self) -> bool:
        return (
            self.resources >= Unit.UNIT_BUILD_COST
            and len(self.units) < ActorConfiguration.UNITS_LIMIT
        )

    def move_unit(self, unit_name: str, position_x: float, position_y: float) -> None:
        unit: Unit = next(filter(lambda unit: unit.name == unit_name, self.units), None)
        unit.move(position_x, position_y)

    def attack(self, unit_name: str, target: arcade.Sprite) -> None:
        unit: Unit = next(filter(lambda unit: unit.name == unit_name, self.units), None)
        unit.attack(target)

    def get_status(self) -> str:
        return f"{self.name}: {self.resources} | {len(self.units)}/{ActorConfiguration.UNITS_LIMIT} | {self.factory.health}/{Factory.BUILDING_BASE_HEALTH}"
