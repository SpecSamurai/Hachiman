from game_configuration import SCREEN_HEIGHT, SCREEN_WIDTH

import sprites.resource as resource

resource_top_left = resource.Resource(
    "resource_top_left",
    SCREEN_WIDTH / 2 - SCREEN_WIDTH / 4,
    SCREEN_HEIGHT / 2 + SCREEN_HEIGHT / 4,
)

resource_bottom_right = resource.Resource(
    "resource_bottom_right",
    SCREEN_WIDTH / 2 + SCREEN_WIDTH / 4,
    SCREEN_HEIGHT / 2 - SCREEN_HEIGHT / 4,
)

resource_center = resource.Resource(
    "resource_center",
    SCREEN_WIDTH / 2,
    SCREEN_HEIGHT / 2,
)

resource_bottom_left = resource.Resource(
    "resource_bottom_left",
    SCREEN_WIDTH / 2 - SCREEN_WIDTH / 4,
    SCREEN_HEIGHT / 2 - SCREEN_HEIGHT / 4,
)

resource_top_right = resource.Resource(
    "resource_top_right",
    SCREEN_WIDTH / 2 + SCREEN_WIDTH / 4,
    SCREEN_HEIGHT / 2 + SCREEN_HEIGHT / 4,
)
