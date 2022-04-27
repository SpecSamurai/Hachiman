from ai.utility_system.actions.attack_by_combat_factory_action import (
    AttackByCombatFactoryAction,
)
from ai.utility_system.actions.attack_by_drone_unit_action import (
    AttackByDroneUnitAction,
)
from ai.utility_system.actions.attack_by_idle_unit_action import AttackByIdleUnitAction
from ai.utility_system.actions.attack_factory_action import AttackFactoryAction
from ai.utility_system.actions.build_action import BuildAction
from ai.utility_system.actions.gather_resources_by_combat_unit_action import (
    GatherResourcesByCombatUnitAction,
)
from ai.utility_system.actions.gather_resources_by_idle_unit_action import (
    GatherResourcesByIdleUnitAction,
)
from ai.utility_system.actions.scoute_action import ScouteAction

BUILD = BuildAction()
GATHER_RESOURCES_BY_IDLE_UNIT = GatherResourcesByIdleUnitAction()
GATHER_RESOURCES_BY_COMBAT_UNIT = GatherResourcesByCombatUnitAction()
ATTACK_BY_IDLE_UNIT = AttackByIdleUnitAction()
ATTACK_BY_DRONE_UNIT = AttackByDroneUnitAction()
ATTACK_FACTORY = AttackFactoryAction()
SCOUTE = ScouteAction()
ATTACK_BY_COMBAT_FACTORY_ACTION = AttackByCombatFactoryAction()
