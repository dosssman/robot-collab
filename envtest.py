import time
import argparse
import logging
import numpy as np

# Robot envs dependencies
from rocobench.envs import SortOneBlockTask, CabinetTask, MoveRopeTask, SweepTask, MakeSandwichTask, PackGroceryTask, MujocoSimEnv, SimRobot, visualize_voxel_scene
from rocobench.envs.task_triage import TriageBlockTask
from rocobench.envs.task_triage_scratch import TriageBlockScratchTask

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
        "triage": TriageBlockTask,
        "triage_scratch": TriageBlockScratchTask
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
        image_hw=(480,480),
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
    t = 0

    from rocobench.envs import SimAction

    print("\n\n### INFO render_cameras")
    print(env.render_cameras)
    # print(env.physics.model.body("ur5e_robotiq"))
    print(env.ndata.qpos)
    print("###\n\n")
    print("\n\n### obs fields")
    # print(obs.objects.keys())
    # env.physics.named.data.qpos[0] = 1.57
    # env.physics.named.data.qpos[1] = 0.785
    env.ndata.qpos[11] = 0.785
    env.physics.forward()
    print("panda qpos and length")
    print(f"Length: {len(obs.panda.qpos)}\n{obs.panda.qpos}")
    print("")
    print("ur5e_robotiq qpos and length")
    print(f"Length: {len(obs.ur5e_robotiq.qpos)}\n{obs.ur5e_robotiq.qpos}")
    print("")
    print("ur5e_suction qpos and length")
    print(f"Length: {len(obs.ur5e_suction.qpos)}\n{obs.ur5e_suction.qpos}")
    print(env.physics.named.data.qpos)
    # print(obs.get_object("panda"))
    print("###\n\n")

    fig, ax = plt.subplots(1, 1)
    # fig, ax = plt.subplots(1, 1, figsize=[6, 6], dpi=400)
    # ax.imshow(env.render_camera(env.render_cameras[-1], width=1500, height=1000))
    # plt.pause(0.1)
    # input()

    # robots_adjust_actions = SimAction(
    #     ctrl_idxs=[i for i in range(17)],
    #     ctrl_vals=[0 for _ in range(17)],

    #     qpos_idxs=[2],
    #     qpos_target=[1.57]
    # )
    # env.step(robots_adjust_actions)
    
    while not done:
        print(f"Current step: {t}")
        dummy_action = SimAction(
            ctrl_idxs=np.arange(17),
            # ctrl_vals=np.random.uniform(0, 0.35, size=[17]),
            ctrl_vals=np.zeros([17]),
            qpos_idxs=np.arange(17),
            qpos_target=np.random.uniform(0, 0.35, size=[17]),
            # qpos_target=np.zeros([17])
        )
        
        _, _, done, _ = env.step(dummy_action)
        ax.imshow(env.render_camera(env.render_cameras[-1], width=480, height=480))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.show(block=False)
        t += 1
        # time.sleep(1)
        plt.pause(0.1)
    
    print(f"Task is done: {done}")
    

    # while True:
    #     env.render()



if __name__ == "__main__":
    main()
