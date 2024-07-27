from typing import Optional, Dict
import os
import json

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
        self.load_value_map()

    def load_value_map(self):
        path = os.path.join(self.save_dir, 'checkpoint_map.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.path_value_map = json.load(f)
        else:
            self.path_value_map = dict()

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None
        if self.monitor_key not in data:
            return None

        value = data[self.monitor_key]
        ckpt_name = self.format_str.format(**data)
        ckpt_path = os.path.join(
            self.save_dir, ckpt_name)
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_name] = value
            self.save_checkpoint_map()
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            self.save_checkpoint_map()

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(os.path.join(self.save_dir, delete_path)):
                os.remove(os.path.join(self.save_dir, delete_path))
            return ckpt_path

    def save_checkpoint_map(self):
        with open(os.path.join(self.save_dir, 'checkpoint_map.json'), 'w') as f:
            json.dump(self.path_value_map, f, indent=4)
