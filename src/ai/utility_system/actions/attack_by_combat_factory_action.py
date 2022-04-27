from __future__ import annotations

import arcade
import ai.utility_system.worldstate as worldstate

from ai.utility_system.actions.baseaction import BaseAction
from sprites.unit import Unit


class AttackByCombatFactoryAction(BaseAction):
    def __init__(self) -> None:
        super().__init__(AttackByCombatFactoryAction.__name__)

    def run_script(self, worldstate: worldstate.WorldState) -> None:
        unit: Unit = next(
            filter(
                lambda unit: unit.status == Unit.COMBAT
                and unit.target == worldstate.enemy.factory,
                worldstate.actor.units,
            ),
            None,
        )
        if unit != None and len(worldstate.actor.enemies_in_sight) > 0:
            enemies_in_sight = worldstate.actor.enemies_in_sight.copy()
            enemies_in_sight.sort(
                key=lambda a: arcade.get_distance_between_sprites(unit, a)
            )
            if len(enemies_in_sight) > 0:
                unit.attack(enemies_in_sight[0])

    def get_score(self, worldstate: worldstate.WorldState) -> int:
        has_combat_unit = any(
            [
                unit
                for unit in worldstate.actor.units
                if unit.status == Unit.COMBAT
                and unit.target == worldstate.enemy.factory
            ]
        )
        enemies_in_sight = len(worldstate.actor.enemies_in_sight) > 0

        score = 4 + 2 * len(worldstate.actor.enemies_in_sight)

        return score if has_combat_unit and enemies_in_sight else 0
