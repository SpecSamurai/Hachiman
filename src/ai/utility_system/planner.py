from ai.utility_system.actions.baseaction import BaseAction
import ai.utility_system.actions.definedactions as actions


class Planner:
    def __init__(self) -> None:
        self.availableActions = [
            actions.ATTACK_BY_DRONE_UNIT,
            actions.ATTACK_BY_IDLE_UNIT,
            actions.ATTACK_FACTORY,
            actions.ATTACK_BY_COMBAT_FACTORY_ACTION,
            actions.BUILD,
            actions.GATHER_RESOURCES_BY_COMBAT_UNIT,
            actions.GATHER_RESOURCES_BY_IDLE_UNIT,
            actions.SCOUTE,
        ]

    def find_action_with_highest_score(self, worldState) -> BaseAction:
        self.availableActions.sort(
            key=lambda action: action.get_score(worldState), reverse=True
        )
        if len(self.availableActions) > 0:
            return self.availableActions[0]
