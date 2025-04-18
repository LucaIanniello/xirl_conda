from typing import Any, Dict, Tuple

import numpy as np
from gym import spaces

import xmagical.entities as en 
from xmagical.entities import EntityIndex
from xmagical.base_env import BaseEnv


DEFAULT_ROBOT_POSE = ((0.0, -0.6), 0.0)
DEFAULT_BLOCK_COLOR = en.ShapeColor.RED
DEFAULT_BLOCK_SHAPE = en.ShapeType.SQUARE
DEFAULT_BLOCK_POSES = [
    ((-0.5, 0.0), 0.0),
    ((0.0, 0.0), 0.0),
    ((0.5, 0.0), 0.0),
]
DEFAULT_GOAL_COLOR = en.ShapeColor.RED
DEFAULT_GOAL_XYHW = (-1.2, 1.16, 0.4, 2.4)
# Max possible L2 distance (arena diagonal 2*sqrt(2)).
D_MAX = 2.8284271247461903


class SweepToTopEnv(BaseEnv):
    """Sweep 3 debris entities to the goal zone at the top of the arena."""

    def __init__(
        self,
        use_state: bool = False,
        use_dense_reward: bool = False,
        use_color_reward: bool = False,
        rand_layout_full: bool = False,
        rand_shapes: bool = False,
        rand_colors: bool = False,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            use_state: Whether to use states rather than pixels for the
                observation space.
            use_dense_reward: Whether to use a dense reward or a sparse one.
            rand_layout_full: Whether to randomize the poses of the debris.
            rand_shapes: Whether to randomize the shapes of the debris.
            rand_colors: Whether to randomize the colors of the debris and the
                goal zone.
        """
        super().__init__(**kwargs)

        self.use_state = use_state
        # self.use_dense_reward = use_dense_reward
        self.use_dense_reward = False
        self.use_color_reward = True
        self.rand_layout_full = rand_layout_full
        self.rand_shapes = rand_shapes
        self.rand_colors = rand_colors
        self.num_debris = 3
        self.stage_completed = [False] * self.num_debris
        self.starting_position = [0] * self.num_debris
        
        

        if self.use_state:
            # Redefine the observation space if we are using states as opposed
            # to pixels.
            c = 4 if self.action_dim == 2 else 5
            self.observation_space = spaces.Box(
                np.array([-1] * (c + 4 * self.num_debris), dtype=np.float32),
                np.array([+1] * (c + 4 * self.num_debris), dtype=np.float32),
                dtype=np.float32,
            )

    def on_reset(self) -> None:
        robot_pos, robot_angle = DEFAULT_ROBOT_POSE
        robot = self._make_robot(robot_pos, robot_angle)

        goal_color = DEFAULT_GOAL_COLOR
        if self.rand_colors:
            goal_color = self.rng.choice(en.SHAPE_COLORS)
        sensor = en.GoalRegion(
            *DEFAULT_GOAL_XYHW,
            goal_color,
            dashed=False,
        )
        self.add_entities([sensor])
        self.__sensor_ref = sensor
        
        # ## RANDOM BLOCKS POSITION
        # robot_x_cord, robot_y_cord = robot_pos
        # goal_x, goal_y, goal_width, goal_height = DEFAULT_GOAL_XYHW
        # space_x_min, space_x_max = -1.1, 1.1
        # space_y_min, space_y_max = -1.1, 1.1 
        
        # x_coords = []
        # y_coords = []
        # while len(x_coords) < self.num_debris:
        #     x = self.rng.uniform(space_x_min, space_x_max)
        #     y = self.rng.uniform(space_y_min, space_y_max)
            
        #     if (abs(x - robot_x_cord) > 0.2 and abs(y - robot_y_cord) > 0.2 and  
        #         not (goal_x <= x <= goal_x + goal_width and goal_y <= y <= goal_y + goal_height)  # Avoid goal
        #        ):
        #         x_coords.append(x)
        #         y_coords.append(y)
        
        # angles = [self.rng.uniform(0, 2 * np.pi) for _ in range(self.num_debris)]
        # debris_shapes = [DEFAULT_BLOCK_SHAPE] * self.num_debris
        # debris_colors = []
        # while len(debris_colors) < self.num_debris:
        #     d_color = self.rng.choice(en.SHAPE_COLORS)
        #     if d_color != goal_color:
        #         debris_colors.append(d_color)
                
        # self.__debris_shapes = [
        #     self._make_shape(
        #     shape_type=shape,
        #     color_name=color,
        #     init_pos=(x, y),
        #     init_angle=angle,
        #     )
        #     for (x, y, angle, shape, color) in zip(
        #     x_coords,
        #     y_coords,
        #     angles,
        #     debris_shapes,
        #     debris_colors,
        #     )
        # ]
        # self.add_entities(self.__debris_shapes)
            
        # Not randomized block positions.
        y_coords = [pose[0][1] for pose in DEFAULT_BLOCK_POSES]
        x_coords = [pose[0][0] for pose in DEFAULT_BLOCK_POSES]
        
        angles = [pose[1] for pose in DEFAULT_BLOCK_POSES]
        # if self.rand_layout_full:
            # The three blocks are located at the same y coordinate but their x
            # coordinate is randomized.
        # y_coord = self.rng.uniform(-0.1, 0.5)
        # y_coords = [y_coord] * 3
        self.starting_position = y_coords
        x_coords = self.rng.choice(
            np.arange(-0.8, 0.8, 4.0 * self.SHAPE_RAD),
            size=self.num_debris,
            replace=False,
        )
        debris_shapes = [DEFAULT_BLOCK_SHAPE] * self.num_debris
        colors_set = [en.ShapeColor.RED, en.ShapeColor.BLUE, en.ShapeColor.YELLOW]
        self.rng.shuffle(colors_set)
        debris_colors = colors_set[: self.num_debris]
        # if self.rand_shapes:
        #     debris_shapes = self.rng.choice(
        #         en.SHAPE_TYPES, size=self.num_debris
        #     ).tolist()
        # if self.rand_colors:
        #     debris_colors = self.rng.choice(
        #         en.SHAPE_COLORS, size=self.num_debris
        #     ).tolist()
        self.__debris_shapes = [
            self._make_shape(
                shape_type=shape,
                color_name=color,
                init_pos=(x, y),
                init_angle=angle,
            )
            for (x, y, angle, shape, color) in zip(
                x_coords,
                y_coords,
                angles,
                debris_shapes,
                debris_colors,
            )
        ]
        self.add_entities(self.__debris_shapes)

        # Add robot last for draw order reasons.
        self.add_entities([robot])

        # Block lookup index.
        self.__ent_index = en.EntityIndex(self.__debris_shapes)
        
        self.stage_completed = [False] * self.num_debris

    def get_state(self) -> np.ndarray:
        robot_pos = self._robot.body.position
        robot_angle_cos = np.cos(self._robot.body.angle)
        robot_angle_sin = np.sin(self._robot.body.angle)
        goal_y = 1
        target_pos = []
        robot_target_dist = []
        target_goal_dist = []
        for target_shape in self.__debris_shapes:
            tpos = target_shape.shape_body.position
            target_pos.extend(tuple(tpos))
            robot_target_dist.append(np.linalg.norm(robot_pos - tpos) / D_MAX)
            gpos = (tpos[0], goal_y)
            target_goal_dist.append(np.linalg.norm(tpos - gpos) / D_MAX)
        state = [
            *tuple(robot_pos),  # 2
            *target_pos,  # 2t
            robot_angle_cos,  # 1
            robot_angle_sin,  # 1
            *robot_target_dist,  # t
            *target_goal_dist,  # t
        ]  # total = 4 + 4t
        if self.action_dim == 3:
            state.append(self._robot.finger_width)

        return np.array(state, dtype=np.float32)

    def score_on_end_of_traj(self) -> float:
        # score = number of debris entirely contained in goal zone / 3
        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            contained=True, ent_index=self.__ent_index
        )
        target_set = set(self.__debris_shapes)
        n_overlap_targets = len(target_set & overlap_ents)
        score = n_overlap_targets / len(target_set)
        if len(overlap_ents) == 0:
            score = 0
        return score

    def _dense_reward(self) -> float:
        """Mean distance of all debris entitity positions to goal zone."""
        y = 1
        target_goal_dists = []
        for target_shape in self.__debris_shapes:
            target_pos = target_shape.shape_body.position
            goal_pos = (target_pos[0], y)  # Top of screen.
            dist = np.linalg.norm(target_pos - goal_pos)
            if target_pos[1] > 0.88:
                dist = 0
            target_goal_dists.append(dist)
        target_goal_dists = np.mean(target_goal_dists)
        return -1.0 * target_goal_dists

    def _sparse_reward(self) -> float:
        """Fraction of debris entities inside goal zone."""
        # `score_on_end_of_traj` is supposed to be called at the end of a
        # trajectory but we use it here since it gives us exactly the reward
        # we're looking for.
        return self.score_on_end_of_traj()

    def _color_reward(self) -> float:
        #Reward function where the robot should move in the goal area the red block, indipendently from its position
        #After the robot should move the blue block and finally the yellow one.
        
        goal_x, goal_y, goal_h, goal_w = DEFAULT_GOAL_XYHW
        goal_y_min = goal_y - goal_h
        goal_y_max = goal_y 
        goal_center_y = (goal_y_min + goal_y_max) / 2

        def in_goal(pos_y):
            return goal_y_min <= pos_y <= goal_y_max

        ##Sequenatial color reward: Red, Blue, Yellow
        red_block = next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.RED)
        red_block_index = next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.RED)
        red_block_starting_y = self.starting_position[red_block_index]

        blue_block = next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.BLUE)
        blue_block_index = next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.BLUE)
        blue_block_starting_y = self.starting_position[blue_block_index]
        
        yellow_block = next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.YELLOW)
        yellow_block_index = next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.YELLOW)
        yellow_block_starting_y = self.starting_position[yellow_block_index]

        red_y = red_block.shape_body.position[1]
        blue_y = blue_block.shape_body.position[1]
        yellow_y = yellow_block.shape_body.position[1]
                
        red_dist = abs(red_y - goal_y_min)
        blue_dist = abs(blue_y - goal_y_min)
        yellow_dist= abs(yellow_y - goal_y_min)
        
        red_dist_init = abs(red_block_starting_y-goal_y_min)
        blue_dist_init = abs(blue_block_starting_y-goal_y_min)
        yellow_dist_init = abs(yellow_block_starting_y - goal_y_min)
        
        reward = 0
        if not self.stage_completed[0]:
            reward += 1.0 / (1.0 + red_dist) - 1.0 / ( 1.0 + red_dist_init)
            if in_goal(red_y):
                self.stage_completed[0] = True
                reward += 1.0
        # elif not self.stage_completed[1]:
        #     reward += 1.0 / (1.0 + blue_dist) - 1.0 / (1.0 + blue_dist_init)
        #     if in_goal(blue_y):
        #         self.stage_completed[1] = True
        #         reward += 1.0
        # elif not self.stage_completed[2]:
        #     reward += 1.0 / (1.0 + yellow_dist) - 1.0 / (1.0 + yellow_dist_init)
        #     if in_goal(yellow_y):
        #         self.stage_completed[2] = True
        #         reward += 1.0
                    
        return reward
            
        
    def get_reward(self) -> float:
        if self.use_dense_reward:
            return self._dense_reward()
        if self.use_color_reward:
            return self._color_reward()
        return self._sparse_reward()

    def reset(self) -> np.ndarray:
        obs = super().reset()
        if self.use_state:
            return self.get_state()
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, rew, done, info = super().step(action)
        if self.use_state:
            obs = self.get_state()
        return obs, rew, done, info
