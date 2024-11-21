from __future__ import annotations

from sheeprl.utils.imports import _IS_CRAFTER_AVAILABLE

if not _IS_CRAFTER_AVAILABLE:
    raise ModuleNotFoundError(_IS_CRAFTER_AVAILABLE)

from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union

import crafter
import crafter.constants as constants

import crafter.objects as objects

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame


class CrafterWrapper(gym.Wrapper):
    
    PLAYER_ID = 13
    ZOMBIE_ID = 15
    SKELETON_ID = 16
    PRESENCE_FILTER = ['water', 'tree', 'lava', 'coal', 'iron', 'diamond', 'table', 'furnace',
                       crafter.objects.Cow, crafter.objects.Zombie, crafter.objects.Skeleton, crafter.objects.Plant]
    
    def __init__(self, id: str, screen_size: Sequence[int, int] | int, seed: int | None = None) -> None:
        assert id in {"crafter_reward", "crafter_nonreward"}
        if isinstance(screen_size, int):
            screen_size = (screen_size,) * 2

        env = crafter.Env(size=screen_size, seed=seed, reward=(id == "crafter_reward"))
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    self.env.observation_space.low,
                    self.env.observation_space.high,
                    self.env.observation_space.shape,
                    self.env.observation_space.dtype,
                )
            }
        )
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self.reward_range = self.env.reward_range or (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        # render
        self._render_mode: str = "rgb_array"
        # metadata
        self._metadata = {"render_fps": 30}
        
        
        # concept supervision
        self.obj2id = self.env._world._mat_ids | self.env._sem_view._obj_ids
        self.id2obj = {v: k for k, v in self.obj2id.items() }
        self.obj_concept_filter = np.array([self.obj2id[concept] for concept in CrafterWrapper.PRESENCE_FILTER])
        self.obj_id = np.array([*self.obj2id.values()])
        
        self.ENTITY_NEAR_THRESHOLD = 8
        self.MATERIAL_PRESENCE_THRESHOLD = 16
        
    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def _can_make(self) -> None:
        make_info = constants.make
        make_concepts = {}
        for item, info in make_info.items():
            # don't require crafting station to fire concept
            # nearby, _ = self.env._world.nearby(self.env._player.pos, 1)
            # if not all(util in nearby for util in info['nearby']):
            #     make_concepts[item] = 0
            #     continue
            if any(self.env._player.inventory[k] < v for k, v in info['uses'].items()):
                make_concepts[item] = 0
                continue
            make_concepts[item] = 1
        
        labels, concepts = list(zip(*make_concepts.items()))    
        
        return np.array(concepts, dtype=np.float32), labels
        
    @staticmethod
    def manhattan(x :  np.ndarray, y : np.ndarray):
        if x.size == 0:
            return np.inf
        return np.abs(x - y).sum(axis=1).min()
        
    def _is_near_hostile(self) -> None:
        
        # player is id = 13, zombie = 15, skele = 16
        view = self.env._sem_view()
        player_pos = np.argwhere(view == CrafterWrapper.PLAYER_ID)[0]
        zombies_pos = np.argwhere(view == CrafterWrapper.ZOMBIE_ID)
        skeles_pos = np.argwhere(view == CrafterWrapper.SKELETON_ID)
        
        # Same as aggro range
        nearby_zombie, nearby_skele = CrafterWrapper.manhattan(zombies_pos, player_pos) <= 3, CrafterWrapper.manhattan(skeles_pos, player_pos) <= 3
        
        return np.array([nearby_zombie, nearby_skele], dtype=np.float32)
                
    def _is_near_material(self) -> None:    
        concept_keys = np.array(self.obj_id[self.obj_concept_filter])
        
        view = self.env._sem_view()
        player_pos = np.argwhere(view == CrafterWrapper.PLAYER_ID)[0]
        
        concept_presences = np.array([CrafterWrapper.manhattan(np.argwhere(view == i), player_pos) for i in concept_keys])
        
        return (concept_presences < self.MATERIAL_PRESENCE_THRESHOLD).astype(np.float32)
        

    def _get_concepts(self, info : Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        info (dict): {'inventory', 'achievements', 'discount', 'semantic', 'player_pos', 'reward'}
        """
        
        # See presence filter for ids        
        material_presence = self._is_near_material()
        # can make [wood_pick, stone_pick, iron_pick, wood_sword, stone_sword, iron_sword]
        can_make, can_make_labels = self._can_make()
        # near [zombie, skeleton]
        near_hostile = self._is_near_hostile()
        # [0, 9] normalized
        player_health = np.array([self.env._player.health / 9])
        # [0, 25] normalized
        player_hunger = np.array([self.env._player._hunger / 25])
        
        print(material_presence, can_make, near_hostile, player_health, player_hunger)
        
        return {"concepts": np.concatenate([material_presence, can_make, near_hostile, player_health, player_hunger])}
        
    def _convert_obs(self, obs: np.ndarray, info : Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {"rgb": obs}, info | self._get_concepts(info)
        
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        obs, info = self._convert_obs(obs, info)
        return obs, reward, done and info["discount"] == 0, done and info["discount"] != 0, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        self.env._seed = seed
        obs = self.env.reset()
        return self._convert_obs(obs, {})

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def close(self) -> None:
        return

concept_dict = {
    'near_water' : 0,
    'near_tree' : 1,
    'near_lava' : 2,
    'near_coal' : 3,
    'near_iron' : 4,
    'near_diamond' : 5,
    'near_table' : 6,
    'near_furnace' : 7,
    crafter.objects.Cow : 8,
    crafter.objects.Zombie : 9,
    crafter.objects.Skeleton : 10,
    crafter.objects.Plant : 11,
    'make_wood_pickaxe' : 12,
    'make_stone_pickaxe' : 13,
    'make_iron_pickaxe' : 14,
    'make_wood_sword' : 15,
    'make_stone_sword' : 16,
    'make_iron_sword': 17,
    'near_zombie': 18,
    'near_skeleton': 19,
    'health': 20,
    'hunger': 21
}