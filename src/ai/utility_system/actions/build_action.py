from __future__ import annotations
import ai.utility_system.worldstate as worldstate

from ai.utility_system.actions.baseaction import BaseAction
from game_configuration import UNIT_BUILD_COST
from sprites.unit import Unit


class BuildAction(BaseAction):
    def __init__(self) -> None:
        super().__init__(BuildAction.__name__)

    def run_script(self, worldstate: worldstate.WorldState) -> None:
        if worldstate.actor.can_build_unit():
            new_unit_name = f"{worldstate.actor.name}_{Unit.__name__}_{len(worldstate.actor.units) + 1}"
            new_unit = worldstate.actor.factory.build_unit(
                new_unit_name, worldstate.actor.configuration.unit_sprite
            )
            worldstate.actor.units.append(new_unit)
            worldstate.actor.resources -= UNIT_BUILD_COST

    def get_score(self, worldstate: worldstate.WorldState) -> int:
        idle_units = len(
            [unit for unit in worldstate.actor.units if unit.status == Unit.IDLE]
        )
        return 10 - idle_units * 10 if worldstate.actor.can_build_unit() else 0
