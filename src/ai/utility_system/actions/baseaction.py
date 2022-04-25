from __future__ import annotations
import actor
import ai.utility_system.worldstate as worldstate


class BaseAction:
    def __init__(self, name: str) -> None:
        self.name = name

    def run_script(self, actor: actor.Actor) -> None:
        pass

    def get_score(self, worldstate: worldstate.WorldState) -> int:
        pass
