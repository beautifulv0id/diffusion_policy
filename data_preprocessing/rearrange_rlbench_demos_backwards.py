import os
from subprocess import call
import pickle
from pathlib import Path
import tap


class Arguments(tap.Tap):
    root_dir: Path

def main(root_dir, task):
    all_variations_path = Path(f'{root_dir}/{task}/all_variations/episodes')

    if not all_variations_path.exists():
        os.makedirs(all_variations_path, exist_ok=True)
    

    variations = os.listdir(f'{root_dir}/{task}/')
    variations = [v for v in variations if (not '.DS_Store' in v) and (not 'all_variations' in v)]

    ep_id = 0
    for variation in variations:
        print(variation)
        var_id = int(variation.replace('variation', ''))

        episodes = os.listdir(f'{root_dir}/{task}/{variation}/episodes')
        episodes = [e for e in episodes if (not '.DS_Store' in e)]
        for episode in episodes:
            call(['ln', '-s',
                  f'{root_dir}/{task}/{variation}/episodes/{episode}',
                  f'{root_dir}/{task}/all_variations/episodes/episode{ep_id}'])
            ep_id += 1

if __name__ == '__main__':
    args = Arguments().parse_args()
    root_dir = str(args.root_dir.absolute())
    tasks = [f for f in os.listdir(root_dir) if ('.zip' not in f) and ('.DS_Store' not in f)]
    for task in tasks:
        print(f'Processing {task}')
        main(root_dir, task)
