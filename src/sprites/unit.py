import os
import math
import arcade


class Unit(arcade.Sprite):
    IDLE = "idle"
    DRONE = "drone"
    COMBAT = "combat"

    UNIT_BUILD_COST = 100
    UNITS_SCALE = 0.5
    UNIT_SPEED = 4.0
    UNIT_BASE_HEALTH = 100.0
    ATTACK_RANGE = 200.0
    ROOT_DIR = os.path.abspath(os.curdir)
    RED_UNIT_SPRITE_NAME = f"{ROOT_DIR}/assets/unit_red.png"
    BLUE_UNIT_SPRITE_NAME = f"{ROOT_DIR}/assets/unit_blue.png"

    def __init__(
        self, name: str, filename: str = None, center_x: float = 0, center_y: float = 0
    ) -> None:
        super().__init__(
            filename=filename,
            scale=Unit.UNITS_SCALE,
            center_x=center_x,
            center_y=center_y,
        )

        self.name: str = name
        self.health: float = Unit.UNIT_BASE_HEALTH

        self.is_moving: bool = False
        self.status = Unit.IDLE

        self.target: Unit = None
        self.target_position_x: float = None
        self.target_position_y: float = None

    def take_damage(self, damage: float) -> None:
        if not self.is_destroyed():
            if self.health - damage > 0:
                self.health -= damage
            else:
                self.health = 0
                super().kill()

    def is_destroyed(self) -> bool:
        return self.health == 0

    def move(self, position_x: float, position_y: float) -> None:
        self.target_position_x = position_x
        self.target_position_y = position_y

        x_diff = position_x - self.center_x
        y_diff = position_y - self.center_y
        angle = math.atan2(y_diff, x_diff)
        self.angle = math.degrees(angle) + 90
        self.change_x = math.cos(angle) * Unit.UNIT_SPEED
        self.change_y = math.sin(angle) * Unit.UNIT_SPEED
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
        self.target_position_x = None
        self.target_position_y = None

    def attack(self, target: arcade.Sprite) -> None:
        self.target = target
        self.status = Unit.COMBAT

    def reached_position(self) -> bool:
        ACCEPTABLE_DISTANCE_FROM_TARGET = 10
        return (
            abs(self.center_x - self.target_position_x)
            < ACCEPTABLE_DISTANCE_FROM_TARGET
            and abs(self.center_y - self.target_position_y)
            < ACCEPTABLE_DISTANCE_FROM_TARGET
        )

    def ready_to_attack(self, target: arcade.Sprite = None) -> bool:
        target = target if target != None else self.target

        if target != None:
            distance = arcade.get_distance_between_sprites(self, self.target)
            return self.status == Unit.COMBAT and distance <= Unit.ATTACK_RANGE
        else:
            return False

    def update(self):
        if self.status == Unit.COMBAT:
            if not self.ready_to_attack():
                self.move(self.target.center_x, self.target.center_y)
            else:
                self.stop()

            if self.target.is_destroyed():
                self.stop()
                self.target = None
                self.status = Unit.IDLE
            else:
                self.point_at_target()

        if self.is_moving:
            if self.target != None:
                self.move(self.target.center_x, self.target.center_y)

            if self.reached_position():
                self.stop()

        super().update()
