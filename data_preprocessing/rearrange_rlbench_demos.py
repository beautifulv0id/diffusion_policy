import os
from subprocess import call
import pickle
from pathlib import Path
import tap


class Arguments(tap.Tap):
    root_dir: Path = Path(os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'peract'))

def main(root_dir, task):
    variations = [ v for v in os.listdir(f'{root_dir}/{task}') if 'variation' in v and 'all_variations' not in v]
    for variation in variations:
        episodes = os.listdir(f'{root_dir}/{task}/{variation}/episodes')
        for episode in episodes:
            # if its a symlink, remove it
            if os.path.islink(f'{root_dir}/{task}/{variation}/episodes/{episode}'):
                os.remove(f'{root_dir}/{task}/{variation}/episodes/{episode}')
        if os.path.islink(f'{root_dir}/{task}/{variation}/variation_descriptions.pkl'):
            os.remove(f'{root_dir}/{task}/{variation}/variation_descriptions.pkl')   
    episodes = os.listdir(f'{root_dir}/{task}/all_variations/episodes')
    episodes = [v for v in episodes if (not '.DS_Store' in v) and (not 'num_objects.txt' in v)]
    seen_variations = {}

    for episode in episodes:
        num = int(episode.replace('episode', ''))
        variation = pickle.load(
            open(
                f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_number.pkl',
                'rb'
            )
        )
        os.makedirs(f'{root_dir}/{task}/variation{variation}/episodes', exist_ok=True)

        if variation not in seen_variations.keys():
            seen_variations[variation] = [num]
        else:
            seen_variations[variation].append(num)

        if os.path.isfile(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl'):
            data1 = pickle.load(open(f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl', 'rb'))
            data2 = pickle.load(open(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl', 'rb'))
            assert data1 == data2
        else:
            call(['ln', '-s',
                  f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl',
                  f'{root_dir}/{task}/variation{variation}/'])

        ep_id = len(seen_variations[variation]) - 1
        # check if link already exists
        if os.path.exists(f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'):
            continue
        call(['ln', '-s',
              "{:s}/{:s}/all_variations/episodes/episode{:d}".format(root_dir, task, num),
              f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'])


if __name__ == '__main__':
    args = Arguments().parse_args()
    root_dir = str(args.root_dir.absolute())
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(root_dir, split)
        tasks = [f for f in os.listdir(split_dir) if ('.zip' not in f) and ('.DS_Store' not in f)]
        for task in tasks:
            print(f'Processing {split_dir}/{task}')
            main(split_dir, task)
