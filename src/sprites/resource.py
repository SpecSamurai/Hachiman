import arcade

class Resource(arcade.SpriteCircle):
    RESOURCE_COLOR_RGB = (253, 120, 54)

    RESOURCE_RADIUS = 50
    RESOURCE_ALPHA = 250

    def __init__(self, name: str, center_x: float = 0, center_y: float = 0) -> None:
        super().__init__(radius=Resource.RESOURCE_RADIUS, color=Resource.RESOURCE_COLOR_RGB, soft=True)

        self.name = name
        self.center_x = center_x
        self.center_y = center_y
        self.alpha = Resource.RESOURCE_ALPHA
        self.drones_count = 0
