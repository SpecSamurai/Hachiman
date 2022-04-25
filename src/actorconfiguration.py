class ActorConfiguration:
    START_RESOURCES = 1000
    UNITS_LIMIT = 20
    DRONES_LIMIT = 10

    def __init__(self, building_sprite: str, unit_sprite: str, position) -> None:
        self.building_sprite = building_sprite
        self.unit_sprite = unit_sprite
        self.position = position
