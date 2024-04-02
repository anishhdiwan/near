import numpy as np
import zarr
import glob


demo_list = []
folders = ['anish', 'gargi', 'tanmay']
ep_ends = []
curr_idx = 0

for folder in folders:
    for np_name in glob.glob(f'data/maze_env/{folder}/*.npy'):
        arr = np.load(np_name)
        demo_list.append(arr)
        curr_idx += len(arr)
        ep_ends.append(curr_idx)


dataset = np.concatenate(demo_list, axis=0)
ep_ends = np.array(ep_ends)

store = zarr.DirectoryStore('data/maze_env/maze_motions.zarr')
root = zarr.group(store=store)
data = root.create_group('data')
meta = root.create_group('meta')
states = data.create_dataset('state', shape=dataset.shape)
ends = meta.create_dataset('episode_ends', shape=ep_ends.shape, dtype=np.intc)

states[:] = dataset
ends[:] = ep_ends

# zarr.save_group('data/maze_env/maze_motions.zarr')