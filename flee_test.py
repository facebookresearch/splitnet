import pdb
import cv2
import numpy as np
import tqdm
from habitat.datasets import make_dataset
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat import SimulatorActions
from reinforcement_learning.get_config import get_dataset_config
from reinforcement_learning.nav_rl_env import RunAwayRLEnv, ExplorationRLEnv

data_subset = "val"
dataset = "suncg"
max_episode_length = 250
RESOLUTION = 10


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


if dataset == "suncg":
    data_path = "data/datasets/pointnav/suncg/v1/{split}/{split}.json.gz"
elif dataset == "mp3d":
    data_path = "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz"
elif dataset == "gibson":
    data_path = "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
else:
    raise NotImplementedError("No rule for this dataset.")

config = get_dataset_config(data_path, data_subset, max_episode_length, 0, [], [])
config.defrost()
for sensor in config.SIMULATOR.AGENT_0.SENSORS:
    config.SIMULATOR[sensor].HEIGHT = RESOLUTION
    config.SIMULATOR[sensor].WIDTH = RESOLUTION
config.TASK.COLLISION_REWARD = 0  # -0.1
config.ENVIRONMENT.MAX_EPISODE_STEPS = 250
config.TASK.TOP_DOWN_MAP.DRAW_SOURCE_AND_TARGET = False
config.TASK.NUM_EPISODES_BEFORE_JUMP = -1
config.TASK.GRID_SIZE = 1
config.TASK.NEW_GRID_CELL_REWARD = 1
config.TASK.RETURN_VISITED_GRID = False

config.freeze()

dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
datasets = {"val": dataset}

env = ExplorationRLEnv(config=config, datasets=datasets)
goal_radius = env.episodes[0].goals[0].radius
if goal_radius is None:
    goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
sim = env.habitat_env.sim
follower = ShortestPathFollower(sim, goal_radius, False)
follower.mode = "geodesic_path"

max_dists = []
for _ in tqdm.tqdm(range(len(env.episodes))):
    env.reset()
    points = []
    dists = []
    agent_state = env.habitat_env.sim.get_agent_state()

    coverages = []
    for trial in range(10):
        env.habitat_env._reset_stats()
        env._visited = set()
        env._check_grid_cell()
        sim._is_episode_active = True
        sim.set_agent_state(position=agent_state.position, rotation=agent_state.rotation)
        next_point = None

        while not env.habitat_env.episode_over:
            while next_point is None:
                point = sim.sample_navigable_point()
                dist = sim.geodesic_distance(agent_state.position, point)
                if np.isfinite(dist):
                    next_point = point
            best_action = follower.get_next_action(
               next_point
            )
            if best_action != SimulatorActions.STOP:
                observations, reward, done, info = env.step(best_action)
            else:
                next_point = None
        coverages.append(len(env._visited))
        print('coverages', max(coverages), coverages)
    max_dists.append(max(coverages))
    print('mean', np.mean(max_dists))



'''
env = RunAwayRLEnv(config=config, datasets=datasets)
goal_radius = env.episodes[0].goals[0].radius
if goal_radius is None:
    goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
sim = env.habitat_env.sim
follower = ShortestPathFollower(sim, goal_radius, False)
follower.mode = "geodesic_path"

max_dists = []
for _ in tqdm.tqdm(range(len(env.episodes))):
    env.reset()
    points = []
    dists = []
    for ii in range(10000):
        agent_state = env.habitat_env.sim.get_agent_state()
        point = sim.sample_navigable_point()
        dist = sim.geodesic_distance(agent_state.position, point)
        points.append(point)
        dists.append(dist)

    order = np.argsort(dists)
    dists = np.array(dists)[order]
    points = np.array(points)[order]
    mask = np.isfinite(dists)
    points = points[mask]
    dists = dists[mask]
    points = points[::-1]
    dists = dists[::-1]
    points = points[::10]
    dists = dists[::10]

    dists_on = []
    for episode, endpoint in enumerate(points[:10]):
        env.habitat_env._reset_stats()
        sim._is_episode_active = True
        sim.set_agent_state(position=agent_state.position, rotation=agent_state.rotation)
        #images = []
        num_steps = 0
        while not env.habitat_env.episode_over:
            num_steps += 1
            best_action = follower.get_next_action(
                endpoint
            )
            observations, reward, done, info = env.step(best_action)
            #im = observations["rgb"].transpose(1, 2, 0)
            #top_down_map = draw_top_down_map(
                #info, observations["heading"], im.shape[0]
            #)
            #output_im = np.concatenate((im, top_down_map), axis=1)
            #images.append(output_im)
        dist_traveled = sim.geodesic_distance(agent_state.position, sim.get_agent_state().position)
        dists_on.append(dist_traveled)
        print('steps', num_steps, 'dist', dist_traveled)

        if num_steps < max_episode_length:
            break
    max_dists.append(max(dists_on))
    print('mean', np.mean(max_dists))
'''
