import math
import arcade

from game_configuration import ATTACK_RANGE, UNIT_BASE_HEALTH, UNIT_SPEED


class Unit(arcade.Sprite):
    IDLE = "idle"
    DRONE = "drone"
    COMBAT = "combat"

    UNITS_SCALE = 0.3

    def __init__(
        self, name: str, filename: str, center_x: float, center_y: float
    ) -> None:
        super().__init__(
            filename=filename,
            scale=Unit.UNITS_SCALE,
            center_x=center_x,
            center_y=center_y,
        )

        self.name: str = name
        self.health: float = UNIT_BASE_HEALTH

        self.is_moving: bool = False
        self.status = Unit.IDLE

        self.target: Unit = None
        self.enemies = 0

    def take_damage(self, damage: float) -> None:
        if not self.is_destroyed():
            if self.health - damage > 0:
                self.health -= damage
            else:
                self.health = 0
                super().kill()

    def is_destroyed(self) -> bool:
        return self.health == 0

    def move_to_target(self) -> None:
        x_diff = self.target.center_x - self.center_x
        y_diff = self.target.center_y - self.center_y
        angle = math.atan2(y_diff, x_diff)
        self.angle = math.degrees(angle) + 90
        self.change_x = math.cos(angle) * UNIT_SPEED
        self.change_y = math.sin(angle) * UNIT_SPEED
        self.is_moving = True

    def gather_resources(self, resource: arcade.Sprite) -> None:
        self.target = resource
        self.is_moving = True
        self.status = Unit.DRONE

    def point_at_target(self) -> None:
        x_diff = self.target.center_x - self.center_x
        y_diff = self.target.center_y - self.center_y
        angle = math.atan2(y_diff, x_diff)
        self.angle = math.degrees(angle) + 90

    def stop(self) -> None:
        self.is_moving = False
        self.change_x = 0
        self.change_y = 0

    def attack(self, target: arcade.Sprite) -> None:
        self.target = target
        self.enemies += 1
        self.status = Unit.COMBAT

    def reached_position(self) -> bool:
        ACCEPTABLE_DISTANCE_FROM_TARGET = 10
        return (
            abs(self.center_x - self.target.center_x) < ACCEPTABLE_DISTANCE_FROM_TARGET
            and abs(self.center_y - self.target.center_y)
            < ACCEPTABLE_DISTANCE_FROM_TARGET
        )

    def ready_to_attack(self) -> bool:
        if self.target != None:
            distance = arcade.get_distance_between_sprites(self, self.target)
            return self.status == Unit.COMBAT and distance <= ATTACK_RANGE
        else:
            return False

    def update(self):
        if self.status == Unit.COMBAT:
            if not self.ready_to_attack():
                self.move_to_target()
            else:
                self.stop()

            if self.target == None or self.target.is_destroyed():
                self.enemies -= 1
                self.stop()
                self.target = None
                self.status = Unit.IDLE
            else:
                self.point_at_target()

        if self.is_moving:
            if self.target != None:
                self.move_to_target()

            if self.reached_position():
                self.stop()

        if (
            self.status == Unit.DRONE
            and self.is_moving
            and self.target != None
            and self.target.drones_count > 5
        ):
            self.stop()
            self.status = Unit.IDLE

        super().update()
