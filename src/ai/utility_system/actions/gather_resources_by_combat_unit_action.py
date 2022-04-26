from __future__ import annotations

import arcade
from actorconfiguration import ActorConfiguration
import ai.utility_system.worldstate as worldstate

from ai.utility_system.actions.baseaction import BaseAction
from sprites.unit import Unit


class GatherResourcesByCombatUnitAction(BaseAction):
    def __init__(self) -> None:
        super().__init__(GatherResourcesByCombatUnitAction.__name__)

    def run_script(self, worldstate: worldstate.WorldState) -> None:
        combat_units = [
            unit for unit in worldstate.actor.units if unit.status == Unit.COMBAT
        ]
        if len(combat_units) > 0:
            distance_to_resources = [
                {
                    "resource": resource,
                    "distance": arcade.get_distance_between_sprites(
                        combat_units[0], resource
                    ),
                }
                for resource in worldstate.resources
                if resource.drones_count < 5
            ]
            distance_to_resources.sort(key=lambda a: a["distance"])
            if len(distance_to_resources) > 0:
                combat_units[0].gather_resources(distance_to_resources[0]["resource"])

    def get_score(self, worldstate: worldstate.WorldState) -> int:
        combat_units_count = len(
            [unit for unit in worldstate.actor.units if unit.status == Unit.COMBAT]
        )
        can_send_more_drones = (
            len([unit for unit in worldstate.actor.units if unit.status == Unit.DRONE])
            < ActorConfiguration.DRONES_LIMIT
        )

        score = combat_units_count - len(worldstate.actor.enemies_in_sight)

        return score if combat_units_count > 0 and can_send_more_drones else 0
