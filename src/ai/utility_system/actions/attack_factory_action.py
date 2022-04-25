from __future__ import annotations

import ai.utility_system.worldstate as worldstate

from ai.utility_system.actions.baseaction import BaseAction
from sprites.unit import Unit


class AttackFactoryAction(BaseAction):
    def __init__(self) -> None:
        super().__init__(AttackFactoryAction.__name__)

    def run_script(self, worldstate: worldstate.WorldState) -> None:
        unit: Unit = next(
            filter(lambda unit: unit.status != Unit.COMBAT, worldstate.actor.units),
            None,
        )
        if unit != None:
            unit.attack(worldstate.enemy.factory)

    def get_score(self, worldstate: worldstate.WorldState) -> int:
        if len(worldstate.actor.units) >= 2 * len(worldstate.enemy.units):
            return 5 + len(worldstate.actor.units) - len(worldstate.enemy.units)
        else:
            return 0
