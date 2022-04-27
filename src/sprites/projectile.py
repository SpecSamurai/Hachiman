import math
import arcade

from actor import Actor
from sprites.unit import Unit


class Projectile(arcade.SpriteCircle):
    PROJECTILE_COLOR_RGB = (255, 197, 37)
    PROJECTILE_RADIUS = 2
    PROJECTILE_SPEED = 8
    PROJECTILE_DAMAGE = 50

    LIFE_SPAN = 25

    def __init__(self, unit: Unit, team: Actor) -> None:
        super().__init__(
            radius=Projectile.PROJECTILE_RADIUS, color=Projectile.PROJECTILE_COLOR_RGB
        )

        self.team = team
        self.center_x = unit.center_x
        self.center_y = unit.center_y
        self.angle = unit.angle

        angle = math.atan2(
            unit.target.center_y - unit.center_y, unit.target.center_x - unit.center_x
        )
        self.change_x = math.cos(angle) * Projectile.PROJECTILE_SPEED
        self.change_y = math.sin(angle) * Projectile.PROJECTILE_SPEED

        self.life_time = 0

    def update(self) -> None:
        if self.life_time > Projectile.LIFE_SPAN:
            super().kill()
        else:
            self.life_time += 1

        super().update()

    def is_frame_valid(frame: int) -> bool:
        return frame % 30 == 0
