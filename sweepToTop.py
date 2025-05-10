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
        colors_set: list = None,
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

        self.use_state = True
        # self.use_dense_reward = use_dense_reward
        self.use_dense_reward = False
        self.use_color_reward = True
        self.rand_layout_full = rand_layout_full
        self.rand_shapes = rand_shapes
        self.rand_colors = rand_colors
        self.num_debris = 3
        self.stage_completed = [False] * self.num_debris
        self.starting_position = [0] * self.num_debris
        self.actual_goal_stage = 0 #0 is red, 1 is blue, 2 is yellow
        self.colors_set = colors_set
        
        

        if self.use_state:
            # Redefine the observation space if we are using states as opposed
            # to pixels.
            # C is the number of states for the robot 
            c = 4 if self.action_dim == 2 else 5
            debris_features = 5
            base_dim = c + debris_features * self.num_debris + 2 * self.num_debris  # robot + (pos+color) + (dist to robot & dist to goal)
            
            goal_dim = self.num_debris  # one-hot index of current goal block

            low = np.array([-1.0] * base_dim + [0.0] * goal_dim, dtype=np.float32)
            high = np.array([+1.0] * base_dim + [1.0] * goal_dim, dtype=np.float32)

            self.observation_space = spaces.Box(low, high, dtype=np.float32)

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
        # x_coords = self.rng.choice(
        #     np.arange(-0.8, 0.8, 4.0 * self.SHAPE_RAD),
        #     size=self.num_debris,
        #     replace=False,
        # )
        debris_shapes = [DEFAULT_BLOCK_SHAPE] * self.num_debris
        
        #self.rng.shuffle(colors_set)
        debris_colors = self.colors_set
        
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
        self.actual_goal_stage = 0
        

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
            color = {
            en.ShapeColor.RED:    [1.0, 0.0, 0.0],
            en.ShapeColor.BLUE:   [0.0, 1.0, 0.0],
            en.ShapeColor.YELLOW: [0.0, 0.0, 1.0],
            }[target_shape.color_name]
            target_pos.extend([tpos[0], tpos[1], *color])
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
        state = np.array(state, dtype=np.float32)
        goal_one_hot = np.zeros(self.num_debris, dtype=np.float32)
        goal_one_hot[self.actual_goal_stage] = 1.0
        return np.concatenate([state, goal_one_hot], axis=0)

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
        """
        Reward function where the robot should move the red block to the goal area first,
        followed by the blue block, and finally the yellow block.
        """
        # Robot position
        robot_pos = np.array([self._robot.body.position.x, self._robot.body.position.y])

        # Goal position
        goal_x, goal_y, goal_h, goal_w = DEFAULT_GOAL_XYHW
        goal_y_min = goal_y - goal_h
        goal_y_max = goal_y
        goal_center_y = (goal_y_min + goal_y_max) / 2
        goal_lower_center = (goal_center_y + goal_y_min) / 2
        
        # Pinch center (robot's gripper center)
        left_finger_body = self._robot.finger_bodies[0]
        right_finger_body = self._robot.finger_bodies[1]
        lf_pos = np.array(left_finger_body.position)
        rf_pos = np.array(right_finger_body.position)
        pinch_center = (lf_pos + rf_pos) / 2.0
        
        # Helper function to check if a block is in the goal area
        def in_goal(pos_y: float) -> bool:
            return goal_lower_center <= pos_y <= goal_y_max

        # Helper function to calculate distances
        def calculate_distances(block, block_starting_y):
            block_x, block_y = block.shape_body.position
            block_pos = np.array([block_x, block_y])
            block_dist_to_goal = abs(block_y - goal_y_min) if (block_y - goal_y_min) <= 0 else 0
            block_dist_to_robot = np.linalg.norm(block_pos - pinch_center)
            block_dist_init = abs(block_starting_y - goal_y_min)
            return block_x, block_y, block_dist_to_goal, block_dist_to_robot, block_dist_init

        # Get blocks and their starting positions
        blocks = {
            "red": next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.RED),
            "blue": next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.BLUE),
            "yellow": next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.YELLOW),
        }
        starting_positions = {
            "red": self.starting_position[next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.RED)],
            "blue": self.starting_position[next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.BLUE)],
            "yellow": self.starting_position[next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.YELLOW)],
        }

        # Calculate distances for each block
        distances = {color: calculate_distances(blocks[color], starting_positions[color]) for color in blocks}

        # Reward calculation
        moving_to_block_reward = 0
        push_reward = 0
        
        if not self.stage_completed[0]:
            # Reward for moving the robot near the red block
            moving_to_block_reward += (1.0 / (1.0 + distances["red"][3]))
            push_reward += (distances["red"][4] - distances["red"][2]) / distances["red"][4]
            if in_goal(distances["red"][1]):
                self.stage_completed[0] = True
                self.actual_goal_stage = 1
        elif not self.stage_completed[1]:
            # Reward for moving the robot near the blue block
            moving_to_block_reward += 1.0 + 1.0 / (1.0 + distances["blue"][3])
            push_reward += 1.0 + (distances["blue"][4] - distances["blue"][2]) / distances["blue"][4]
            if in_goal(distances["blue"][1]):
                self.stage_completed[1] = True
                self.actual_goal_stage = 2
        elif not self.stage_completed[2]:
            # Reward for moving the robot near the yellow block
            moving_to_block_reward += 2.0 + 1.0 / (1.0 + distances["yellow"][3])
            push_reward += 2.0 + (distances["yellow"][4] - distances["yellow"][2]) / distances["yellow"][4]
            if in_goal(distances["yellow"][1]):
                self.stage_completed[2] = True


        reward = 0.3 * moving_to_block_reward + 0.7 * push_reward
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
