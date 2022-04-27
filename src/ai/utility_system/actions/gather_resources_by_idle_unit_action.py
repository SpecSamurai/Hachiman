from __future__ import annotations

import arcade
import ai.utility_system.worldstate as worldstate

from ai.utility_system.actions.baseaction import BaseAction
from game_configuration import DRONES_LIMIT
from sprites.unit import Unit


class GatherResourcesByIdleUnitAction(BaseAction):
    def __init__(self) -> None:
        super().__init__(GatherResourcesByIdleUnitAction.__name__)

    def run_script(self, worldstate: worldstate.WorldState) -> None:
        idle_units = [
            unit for unit in worldstate.actor.units if unit.status == Unit.IDLE
        ]
        if len(idle_units) > 0:
            distance_to_resources = [resource for resource in worldstate.resources if resource.drones_count < 5]
            distance_to_resources.sort(key=lambda a: arcade.get_distance_between_sprites(idle_units[0], a))
            if len(distance_to_resources) > 0:
                idle_units[0].gather_resources(distance_to_resources[0])

    def get_score(self, worldstate: worldstate.WorldState) -> int:
        has_idle_unit = any(
            [unit for unit in worldstate.actor.units if unit.status == Unit.IDLE]
        )

        can_send_more_drones = (
            len([unit for unit in worldstate.actor.units if unit.status == Unit.DRONE])
            < DRONES_LIMIT
        )

        return (
            10 - len(worldstate.actor.enemies_in_sight)
            if has_idle_unit and can_send_more_drones
            else 0
        )
