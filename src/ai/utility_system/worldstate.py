import arcade
from actor import Actor
from game_configuration import ATTACK_RANGE


class WorldState:
    def __init__(self, actor: Actor, enemy: Actor, resources) -> None:
        actor.enemies_in_sight.clear()
        for unit in actor.units:
            for enemy_unit in enemy.units:
                distance = arcade.get_distance_between_sprites(unit, enemy_unit)
                if distance < ATTACK_RANGE:
                    actor.enemies_in_sight.append(enemy_unit)

        self.actor: Actor = actor
        self.enemy: Actor = enemy
        self.resources = resources
