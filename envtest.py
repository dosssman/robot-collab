import time
import argparse
import logging
import numpy as np

# Robot envs dependencies
from rocobench.envs import SortOneBlockTask, CabinetTask, MoveRopeTask, SweepTask, MakeSandwichTask, PackGroceryTask, MujocoSimEnv, SimRobot, visualize_voxel_scene

# Plotting and visualization of the observations
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="sort_one") # "sort_one" not in that TASK_NAME_MAP directory
    parser.add_argument("--task", type=str, default="pack")

    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--temperature", "-temp", type=float, default=0)
    parser.add_argument("--start_id", "-sid", type=int, default=-1)
    parser.add_argument("--num_runs", '-nruns', type=int, default=1)
    parser.add_argument("--run_name", "-rn", type=str, default="test")
    parser.add_argument("--tsteps", "-t", type=int, default=10)
    parser.add_argument("--output_mode", type=str, default="action_only", choices=["action_only", "action_and_path"])
    parser.add_argument("--comm_mode", type=str, default="dialog", choices=["chat", "plan", "dialog"])
    parser.add_argument("--control_freq", "-cf", type=int, default=15)
    parser.add_argument("--skip_display", "-sd", action="store_true")
    parser.add_argument("--direct_waypoints", "-dw", type=int, default=5)
    parser.add_argument("--num_replans", "-nr", type=int, default=5)
    parser.add_argument("--cont", "-c", action="store_true")
    parser.add_argument("--load_run_name", "-lr", type=str, default="sort_task")
    parser.add_argument("--load_run_id", "-ld", type=int, default=0)
    parser.add_argument("--max_failed_waypoints", "-max", type=int, default=1)
    parser.add_argument("--debug_mode", "-i", action="store_true")
    parser.add_argument("--use_weld", "-w", type=int, default=1)
    parser.add_argument("--rel_pose", "-rp", action="store_true")
    parser.add_argument("--split_parsed_plans", "-sp", action="store_true")
    parser.add_argument("--no_history", "-nh", action="store_true")
    parser.add_argument("--no_feedback", "-nf", action="store_true")
    parser.add_argument("--llm_source", "-llm", type=str, default="gpt-4")
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    TASK_NAME_MAP = {
        "sort": SortOneBlockTask,
        "cabinet": CabinetTask,
        "rope": MoveRopeTask,
        "sweep": SweepTask,
        "sandwich": MakeSandwichTask,
        "pack": PackGroceryTask,
    }

    assert args.task in TASK_NAME_MAP.keys(), f"Task {args.task} not supported"
    env_cl = TASK_NAME_MAP[args.task]

    if args.task == 'rope':
        args.output_mode = 'action_and_path'
        args.split_parsed_plans = True
        logging.warning("MoveRopeTask requires split parsed plans\n")

        args.control_freq = 20
        args.max_failed_waypoints = 0
        logging.warning("MopeRope requires max failed waypoints 0\n")
        if not args.no_feedback:
            args.tstep = 5
            logging.warning("MoveRope needs only 5 tsteps\n")

    elif args.task == 'pack':
        args.output_mode = 'action_and_path'
        args.control_freq = 10
        args.split_parsed_plans = True
        args.max_failed_waypoints = 0
        args.direct_waypoints = 0
        logging.warning("PackGroceryTask requires split parsed plans, and no failed waypoints, no direct waypoints\n")

    render_freq = 600
    if args.control_freq == 15:
        render_freq = 1200
    elif args.control_freq == 10:
        render_freq = 2000
    elif args.control_freq == 5:
        render_freq = 3000
    env = env_cl(
        render_freq=render_freq,
        image_hw=(400,400),
        sim_forward_steps=300,
        error_freq=30,
        error_threshold=1e-5,
        randomize_init=True,
        render_point_cloud=0,
        render_cameras=["face_panda","face_ur5e","teaser",],
        # one_obj_each=True, # rocobench.envs.base_env.py's MujocoSimEnv does not seem to support this argument.
    )
    print(env)
    # robots = env.get_sim_robots()
    # print(robots)

    # reset env ?
    obs = env.reset()
    done = False

    from rocobench.envs import SimAction
    dummy_action = SimAction(
        ctrl_idxs=np.arange(17),
        ctrl_vals=np.random.uniform(0, 0.1, size=[17]),
        qpos_idxs=np.arange(17),
        qpos_target=np.random.uniform(0, 0.1, size=[17])
    )

    while not done:
        _, _, done, _ = env.step(dummy_action)
        plt.imshow(env.render_camera(env.render_cameras[-1]))
        plt.show(block=False)
        # time.sleep(1)
        plt.pause(0.1)
    
    print(obs)

    # while True:
    #     env.render()



if __name__ == "__main__":
    main()
