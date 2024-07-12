from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.env.rlbench.rlbench_utils import task_file_to_task_class, Actioner, Mover, get_actions_from_demo
from diffusion_policy.common.rlbench_util import extract_obs
import torch
import numpy as np
from rlbench.task_environment import TaskEnvironment
from rlbench.demo import Demo
from typing import List
from diffusion_policy.common.rlbench_util import extract_obs, create_obs_state_plot
from diffusion_policy.common.pytorch_util import dict_apply
import torch.nn.functional as F
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from multiprocessing import Process, Manager


@torch.no_grad()
def _evaluate_task_on_demos(env_args : dict, 
                            task_str: str,
                            demos: List[Demo],  
                            max_steps: int,
                            actioner: Actioner,
                            n_procs_max : int = 1, 
                            max_rrt_tries: int = 1,
                            demo_tries: int = 1,
                            n_visualize: int = 0,
                            verbose: bool = False,
                            plot_gt_action: bool = False,
                            return_model_obs : bool = False):
    n_procs = min(n_procs_max, len(demos))

    if n_procs == 1:
        proc_log_data = [{}]
        _evaluate_task_on_demos_multiproc(0, 1, proc_log_data, env_args, task_str, demos, 
                                          max_steps, actioner, max_rrt_tries, demo_tries, 
                                          n_visualize, verbose, plot_gt_action=plot_gt_action,
                                          return_model_obs=return_model_obs)
    else:
        manager = Manager()
        proc_log_data = manager.dict()
        processes = [Process(target=_evaluate_task_on_demos_multiproc, 
                                        args=(i, n_procs, proc_log_data, env_args, 
                                            task_str, demos, max_steps, actioner, 
                                            max_rrt_tries, demo_tries, n_visualize, 
                                            verbose, plot_gt_action,return_model_obs)
                            ) for i in range(n_procs)]
        [p.start() for p in processes]
        [p.join() for p in processes]

    log_data = {k: [] for k in proc_log_data[0].keys()}
    for i in range(len(proc_log_data)):
        proc_data = proc_log_data[i]
        for k, v in proc_data.items():
            if type(v) == list:
                log_data[k].extend(v)
            else:
                log_data[k].append(v)
    
    successful_demos = sum(log_data["successful_demos"])
    success_rate = successful_demos / (len(demos) * demo_tries)
    log_data["success_rate"] = success_rate
    log_data.pop("successful_demos")
    return log_data

@torch.no_grad()
def _evaluate_task_on_demos_multiproc(proc_num : int, 
                                      num_procs : int,
                            proc_log_data : List[dict],
                            env_args : dict,      
                            task_str: str,
                            demos: List[Demo],  
                            max_steps: int,
                            actioner: Actioner,
                            max_rrt_tries: int = 1,
                            demo_tries: int = 1,
                            n_visualize: int = 0,
                            verbose: bool = False,
                            plot_gt_action: bool = False,
                            return_model_obs: bool = False):
    env = RLBenchEnv(**env_args)
    env.launch()
    device = actioner.device
    dtype = actioner.dtype
    successful_demos = 0
    total_reward = 0
    log_data = {
        "rgbs" : [],
        "obs_state" : [],
        "mask" : [],
    }
    if return_model_obs:
        log_data["model_obs"] = []

    task_type = task_file_to_task_class(task_str)
    task : TaskEnvironment = env.env.get_task(task_type)
    task.set_variation(0)


    n_obs_steps = env.n_obs_steps
    
    for demo_id in range(proc_num, len(demos), num_procs):
        demo = demos[demo_id]
        if plot_gt_action:
            gt_actions = get_actions_from_demo(demo)
        gt_action = None

        if verbose:
                print()
                print(f"Starting demo {demo_id}")
        for demo_try_i in range(demo_tries):
            if demo_id < n_visualize and demo_try_i == 0:
                env.start_recording()
                obs_state = []
                if env.apply_mask:
                    logging_masks = []
                if return_model_obs:
                    model_obs = []

            rgbs = torch.Tensor([])
            pcds = torch.Tensor([])
            masks = torch.Tensor([])
            low_dim_pcds = torch.Tensor([])
            keypoint_poses = torch.Tensor([])
            grippers = torch.Tensor([])
            low_dim_states = torch.Tensor([])
            env.reset_obs_history()

            descriptions, observation = task.reset_to_demo(demo)
            env.store_obs(observation)
            env.record_frame(observation)
            actioner.load_episode(task_str, demo.variation_number)

            move = Mover(task, max_tries=max_rrt_tries)
            reward = 0.0
            max_reward = 0.0

            for step_id in range(max_steps):
                # Fetch the current observation, and predict one action
                for obs in env.get_obs_history():
                    obs_dict = extract_obs(obs,
                                            cameras=env.apply_cameras,
                                            use_rgb=env.apply_rgb,
                                            use_pcd=env.apply_pc,
                                            use_mask=env.apply_mask,
                                            use_pose=env.apply_poses,
                                            use_low_dim_state=True, 
                                            use_low_dim_pcd=env.apply_low_dim_pcd)
                    obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))

                    if env.apply_rgb:
                        rgb = obs_dict['rgb']
                        rgbs = torch.cat([rgbs, rgb], dim=0)
                    if env.apply_pc:
                        pcd = obs_dict['pcd']
                        pcds = torch.cat([pcds, pcd], dim=0)
                    if env.apply_low_dim_pcd:
                        low_dim_pcd = obs_dict['low_dim_pcd']
                        low_dim_pcds = torch.cat([low_dim_pcds, low_dim_pcd], dim=0)
                    if env.apply_poses:
                        keypoint_pose = obs_dict['keypoint_poses']
                        keypoint_poses = torch.cat([keypoint_poses, keypoint_pose], dim=0)
                    if env.apply_mask:
                        mask = obs_dict['mask']
                        masks = torch.cat([masks, mask], dim=0)

                    gripper = obs_dict['curr_gripper']
                    grippers = torch.cat([grippers, gripper], dim=0)
                    low_dim_state = obs_dict['low_dim_state']
                    low_dim_states = torch.cat([low_dim_states, low_dim_state], dim=0)
                
                def pad_input(input : torch.Tensor, npad):
                    sh_in = input.shape
                    input = input[-n_obs_steps:].unsqueeze(0)
                    input = input.reshape(input.shape[:2] + (-1,))
                    input = F.pad(
                        input, (0, 0, npad, 0), mode='replicate'
                    )
                    input = input.view((1, n_obs_steps, ) + sh_in[1:])
                    return input

                # Prepare proprioception history
                npad = n_obs_steps - grippers[-n_obs_steps:].shape[0]
                obs_dict = dict()
                obs_dict["curr_gripper"] = pad_input(grippers, npad)
                grippers = grippers[-n_obs_steps:]
                obs_dict["low_dim_state"] = pad_input(low_dim_states, npad)
                low_dim_states = low_dim_states[-n_obs_steps:]

                if env.apply_rgb:
                    obs_dict["rgb"] = rgbs[-1:]
                    rgbs = rgbs[-n_obs_steps:]
                if env.apply_pc:
                    obs_dict["pcd"]  = pcds[-1:]
                    pcds = pcds[-n_obs_steps:]
                if env.apply_low_dim_pcd:
                    obs_dict["low_dim_pcd"] = low_dim_pcds[-1:]
                    low_dim_pcds = low_dim_pcds[-n_obs_steps:]
                if env.apply_poses:
                    obs_dict["keypoint_poses"] = keypoint_poses[-1:]
                    keypoint_poses = keypoint_poses[-n_obs_steps:]
                if env.apply_mask:
                    obs_dict["mask"] = masks[-1:].bool()
                    masks = masks[-n_obs_steps:]


                out = actioner.predict(dict_apply(obs_dict, lambda x: x.type(dtype).to(device)))
                trajectory = out['rlbench_action']

                if plot_gt_action:
                    gt_action = gt_actions[step_id][None]

                if env._recording:
                    obs_state.append(create_obs_state_plot(obs_dict, lowdim=env.apply_low_dim_pcd, pred_action=torch.from_numpy(trajectory)[None], gt_action=gt_action)[0])
                    if env.apply_mask:
                        obs_state.append(create_obs_state_plot(obs_dict, use_mask=True, pred_action=torch.from_numpy(trajectory)[None], gt_action=gt_action)[0])
                    if env.apply_mask:
                        logging_masks.append((masks[-1,-1].int() * 255).expand(3, -1, -1).cpu().numpy().astype(np.uint8))
                
                if return_model_obs:
                    model_obs.append(obs_dict)
                    
                if verbose:
                    print(f"Step {step_id}")

                terminate = True

                # Update the observation based on the predicted action
                try:
                    if verbose:
                        print("Plan with RRT")

                    for action in trajectory[:env.n_action_steps]:
                        collision_checking = env._collision_checking(task_str, step_id)
                        observation, reward, terminate, _ = move(action, collision_checking=collision_checking)

                    env.store_obs(observation)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        successful_demos += 1
                        break

                    if terminate:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, successful_demos, e)
                    reward = 0
                    #break
            if demo_id < n_visualize and demo_try_i == 0:
                env.stop_recording()
                rgbs = env.get_rgbs()
                log_data["rgbs"].append(rgbs)
                obs_state = np.array(obs_state)
                log_data["obs_state"].append(obs_state)
                if return_model_obs:
                    log_data["model_obs"].append(model_obs)
                if env.apply_mask:
                    masks = np.array(logging_masks)
                    log_data["mask"].append(masks)

            total_reward += max_reward

            print(
                task_str,
                "Variation",
                demo.variation_number,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {successful_demos / ((demo_id+1) * demo_tries)}",
                f"Total reward: {total_reward:.2f}/{(demo_id+1) * demo_tries}"                )

    log_data.update( {
        "successful_demos": successful_demos,
    })
    env.shutdown()
    proc_log_data[proc_num] = log_data
