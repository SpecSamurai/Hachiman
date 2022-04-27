import arcade
from game_configuration import (
    ATTACK_RANGE,
    BACKGROUND_COLOR_RGB,
    GREEN_COLOR_RGB,
    GREEN_POSITION,
    GREEN_UNIT_SPRITE_NAME,
    RED_COLOR_RGB,
    RED_POSITION,
    RED_UNIT_SPRITE_NAME,
    SCREEN_HEIGHT,
    SCREEN_TITLE,
    SCREEN_WIDTH,
)

import sprites.resources as resources

from actorconfiguration import ActorConfiguration
from ai.utility_system.worldstate import WorldState
from sprites.factory import Factory
from sprites.projectile import Projectile
from sprites.unit import Unit
from actor import Actor


class GameWindow(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.debug = False

        arcade.set_background_color(BACKGROUND_COLOR_RGB)
        self.set_update_rate(1 / 30)

        self.frame_count = 0

        self.red_actor: Actor = None
        self.green_actor: Actor = None

        self.buildings: arcade.SpriteList = None
        self.resources: arcade.SpriteList = None
        self.projectiles: arcade.SpriteList = None

    def add_resources(self):
        self.resources = arcade.SpriteList()
        self.resources.append(resources.resource_top_left)
        self.resources.append(resources.resource_bottom_right)
        self.resources.append(resources.resource_center)
        self.resources.append(resources.resource_bottom_left)
        self.resources.append(resources.resource_top_right)

    def setup(self):
        self.add_resources()
        self.projectiles = arcade.SpriteList()

        self.red_actor = Actor(
            "RED",
            ActorConfiguration(RED_COLOR_RGB, RED_UNIT_SPRITE_NAME, RED_POSITION),
        )

        self.green_actor = Actor(
            "GREEN",
            ActorConfiguration(GREEN_COLOR_RGB, GREEN_UNIT_SPRITE_NAME, GREEN_POSITION),
        )

        self.buildings = arcade.SpriteList()
        self.buildings.append(self.red_actor.factory)
        self.buildings.append(self.green_actor.factory)

    def on_draw(self):
        self.clear()

        self.resources.draw()
        self.projectiles.draw()

        self.green_actor.factory.draw()
        self.red_actor.factory.draw()

        self.red_actor.units.draw()
        self.green_actor.units.draw()

        arcade.draw_text(self.red_actor.get_status(), 10, 10, font_size=12)
        arcade.draw_text(self.green_actor.get_status(), 10, 30, font_size=12)

        if self.debug:
            for unit in self.red_actor.units:
                arcade.draw_circle_outline(
                    unit.center_x, unit.center_y, ATTACK_RANGE, arcade.color.BLUE
                )

    def on_update(self, delta_time):
        self.buildings.update_animation()

        if self.red_actor.does_plan_exist():
            self.red_actor.perform_plan(
                WorldState(self.red_actor, self.green_actor, self.resources)
            )
        else:
            self.red_actor.replan(
                WorldState(self.red_actor, self.green_actor, self.resources)
            )

        if self.green_actor.does_plan_exist():
            self.green_actor.perform_plan(
                WorldState(self.green_actor, self.red_actor, self.resources)
            ),
        else:
            self.green_actor.replan(
                WorldState(self.green_actor, self.red_actor, self.resources)
            )

        self.update_units()
        self.update_projectiles()

    def update_units(self) -> None:
        self.frame_count += 1

        for reso in self.resources:
            collisions = arcade.check_for_collision_with_lists(
                reso, [self.red_actor.units, self.green_actor.units]
            )

            reso.drones_count = len(
                [unit for unit in self.red_actor.units if unit.target == reso]
            ) + len([unit for unit in self.green_actor.units if unit.target == reso])

        unit: Unit
        for unit in self.red_actor.units:
            if unit.status == Unit.DRONE and self.frame_count % 15 == 0:
                collisions = arcade.check_for_collision_with_list(unit, self.resources)

                for _ in collisions:
                    self.red_actor.resources += 1

            self.red_actor.units_collisions_update(unit)

            if unit.ready_to_attack() and Projectile.is_frame_valid(self.frame_count):
                projectile = Projectile(unit, self.red_actor)
                self.projectiles.append(projectile)

        for unit in self.green_actor.units:
            if unit.status == Unit.DRONE and self.frame_count % 15 == 0:
                collisions = arcade.check_for_collision_with_list(unit, self.resources)
                for _ in collisions:
                    self.green_actor.resources += 1

            self.green_actor.units_collisions_update(unit)

            if unit.ready_to_attack() and Projectile.is_frame_valid(self.frame_count):
                projectile = Projectile(unit, self.green_actor)
                self.projectiles.append(projectile)

        self.green_actor.units.update()
        self.red_actor.units.update()

    def update_projectiles(self) -> None:
        projectile: Projectile
        for projectile in self.projectiles:
            hit_list = arcade.check_for_collision_with_lists(
                projectile,
                [
                    self.green_actor.units
                    if projectile.team == self.red_actor
                    else self.red_actor.units,
                    self.buildings,
                ],
            )

            if len(hit_list) > 0:
                projectile.remove_from_sprite_lists()
                for sprite in hit_list:
                    sprite.take_damage(Projectile.PROJECTILE_DAMAGE)

            if not self.is_point_inside_window(
                (projectile.center_x, projectile.center_y)
            ):
                projectile.remove_from_sprite_lists()

        self.projectiles.update()

    def is_point_inside_window(self, point: tuple[float, float]) -> bool:
        x, y = point
        return x > 0 and x < SCREEN_WIDTH and y > 0 and y < SCREEN_HEIGHT

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.D:
            self.debug = not self.debug
