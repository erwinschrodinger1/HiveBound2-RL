import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .globals import *
from .engine import *
from .characters import *

FREE_CAMERA = False


class HiveBoundEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render: bool = False, reset_after_no_improvements: int = 20):
        super().__init__()

        self.width = WIN_WIDTH
        self.height = WIN_HEIGHT
        self.fps = FPS
        self.render_mode = "human" if render else None
        self.time_limit_for_reset_if_no_change_seconds = reset_after_no_improvements

        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.SRCALPHA
        )

        self.game_surface = pygame.Surface(
            (SURFACE_WIDTH, SURFACE_HEIGHT), pygame.SRCALPHA
        )
        self.game_surface.set_alpha(100)

        self.light_surface = pygame.Surface(
            (SURFACE_WIDTH, SURFACE_HEIGHT), pygame.SRCALPHA
        )
        self.light_surface.set_colorkey((0, 0, 0))

        self.checkpoint = (100, 1800 - 64)
        self.player = Player(Sprite.player_sprite, (100, 1800 - 32))
        self.player.checkpoint = self.checkpoint

        self.left = self.right = self.up = self.down = False

        # Loading map
        with open("./assets/map/map.json", "r") as f:
            self.map = json.load(f)
            rects = []
            for r in self.map["rects"]:
                rects.append(pygame.Rect(r[0], r[1], r[2], r[3]))
            self.map["rects"] = rects

        self.map_sprite = SpriteSheet(self.map["image"])
        self.map_img = self.map_sprite.image_at(0, 0, 300, 1800)

        # Loading up guards
        guard_pos = self.map["guard_pos"]
        self.guards = []
        for i in guard_pos:
            pos = guard_pos[i]["pos"]
            delay = guard_pos[i]["delay"]
            f = guard_pos[i]["f"]
            self.guards.append(
                Guard(Sprite.guard_sprite, (pos[0], pos[1] - 32), delay, f, 30, 200)
            )

        self.guard = Guard(Sprite.guard_sprite, (0, 1500 - 128), 5, 2, 30, 200)

        self.last_time = time.time()
        self.last_best_position = self.checkpoint  # initial player position
        self.last_best_position_change_time = time.time()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(5)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([WIN_WIDTH, 1800]),
            dtype=np.int64,
        )

    def step(self, action):

        self.player.event(action)

        match action:
            case 0:
                self.up = True
            case 1:
                self.left = True
            case 2:
                self.down = True
            case 3:
                self.right = True
            case _:
                self.up = False
                self.left = False
                self.down = False
                self.right = False

        self.observation = np.array(
            [self.player.rect.centerx, self.player.rect.centery]
        )
        self.reward = -self.player.rect.centery

        self.terminated = self.player.rect.y < 50

        truncated = False

        current_time = time.time()
        if self.player.rect.centery < self.last_best_position[1]:
            self.last_best_position_change_time = current_time
            self.last_best_position = (
                self.player.rect.centerx,
                self.player.rect.centery,
            )
        else:
            if (
                current_time - self.last_best_position_change_time
                > self.time_limit_for_reset_if_no_change_seconds
            ):
                truncated = True
                self.last_best_position_change_time = current_time

        if self.terminated or truncated:
            self.reset()

        info = {}
        return self.observation, self.reward, self.terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False

        self.camera = [0, 0]
        self.player.jump_to_checkpoint()

        self.running = True
        self.clock = pygame.time.Clock()

        self.surface = self.screen

        self.obeservation = np.array(
            [self.player.rect.centerx, self.player.rect.centery]
        )

        info = {}

        return self.obeservation, info

    def render(self):

        if self.render is None:
            return

        dt = time.time() - self.last_time
        dt *= self.fps
        self.last_time = time.time()

        self.game_surface.fill((0, 0, 0))
        self.light_surface.fill((0, 0, 0))

        if FREE_CAMERA:
            if self.left:
                self.camera[0] -= 5
            if self.right:
                self.camera[0] += 5
            if self.up:
                self.camera[1] -= 5
            if self.down:
                self.camera[1] += 5
        else:
            self.camera[0] += (
                self.player.rect.x - self.camera[0] - SURFACE_WIDTH / 2
            ) / 10
            self.camera[1] += (
                self.player.rect.y - self.camera[1] - SURFACE_HEIGHT / 1.5
            ) / 10

        self.game_surface.blit(self.map_img, (-self.camera[0], -self.camera[1]))

        for guard in self.guards:
            guard.update(self.game_surface, self.light_surface, dt, self.camera)
            if guard.detect_target(
                pygame.Rect(
                    self.player.rect.x - self.camera[0],
                    self.player.rect.y - self.camera[1],
                    self.player.rect.w,
                    self.player.rect.h,
                ),
                self.camera,
            ):
                self.player.jump_to_checkpoint()

        self.player.update(self.game_surface, self.map["rects"], dt, self.camera)

        # Drawing white screen when reached at home
        if self.player.rect.y <= 250:
            pygame.draw.rect(
                self.light_surface,
                [255, 255, 255, 255 - self.player.rect.y],
                (0, 0, self.light_surface.get_width(), self.light_surface.get_height()),
            )

        self.surface.blit(
            pygame.transform.scale(
                self.game_surface, (self.surface.get_width(), self.surface.get_height())
            ),
            (0, 0),
        )
        self.surface.blit(
            pygame.transform.scale(
                self.light_surface,
                (self.surface.get_width(), self.surface.get_height()),
            ),
            (0, 0),
        )

        pygame.display.update()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
