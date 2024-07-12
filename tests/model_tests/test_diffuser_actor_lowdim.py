from diffusion_policy.policy.diffuser_actor_lowdim import DiffuserActor
from diffusion_policy.dataset.rlbench_dataset import RLBenchLowDimNextBestPoseDataset
from diffusion_policy.common.pytorch_util import print_dict
import os
from diffusion_policy.common.rlbench_util import get_task_num_low_dim_pcd, get_gripper_loc_bounds
from absl import app
from absl import flags
import torch


FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/output/test/diffuser_actor_lowdim',
                    'Where to save the dataset.')
flags.DEFINE_string('data_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/data/lowdim_keypoints.zarr',
                    'Path to the data folder.')
flags.DEFINE_integer('n_demos', -1, 'Number of demos to use.')
flags.DEFINE_list('tasks', ['put_item_in_drawer', 'open_drawer_keypoint', 'stack_blocks'], 'Tasks to use.')
flags.DEFINE_string('num_objects_path', 
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/diffusion_policy/tasks/peract_tasks_num_low_dim_pcd.json', 
                    'Path to the number of objects in each task.')
flags.DEFINE_string('tasks_location_bounds_path', 
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/diffusion_policy/tasks/18_peract_tasks_location_bounds.json', 
                    'Path to the number of objects in each task.')


def main(argv):
    task = FLAGS.tasks[0]
    dataset = RLBenchLowDimNextBestPoseDataset(
        dataset_path=FLAGS.data_path,
        task_name=task,
        n_episodes=-1,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    npcd = get_task_num_low_dim_pcd(FLAGS.num_objects_path, task)
    gripper_loc_bounds = get_gripper_loc_bounds(FLAGS.tasks_location_bounds_path, task=task)

    actor = DiffuserActor(
        embedding_dim=192,
        gripper_loc_bounds=gripper_loc_bounds,
        nkeypoints=npcd,
        nhorizon=1,
    )

    batch = next(iter(dataloader))
    loss = actor.compute_loss(batch)
    print("Loss:", loss.cpu().item())

    with torch.no_grad():
        output = actor.predict_action(batch['obs'])
    print_dict(output)

    with torch.no_grad():
        output = actor.evaluate(batch)
    print_dict(output)


if __name__ == '__main__':
  app.run(main)
