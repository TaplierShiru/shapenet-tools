import numpy as np
import os
import h5py
from PIL import Image
import argparse

from tqdm import tqdm


def main(args):
    os.makedirs(args.save_folder, exist_ok=True)
    for category in os.listdir(args.base_dir):
        print(f'Prepare dataset with category {category}')
        render_data_path = os.path.join(args.base_dir, category)
        if not os.path.isdir(render_data_path):
            print(f'Part of the dataset not found!')
            continue
            
        hdf5_path = os.path.join(args.save_folder, f'{category}_render.hdf5')

        render_folder_path_list = os.listdir(render_data_path)
        render_folder_path_list = sorted(
            render_folder_path_list,
            key=lambda x: os.path.relpath(x, render_data_path).split(os.sep)[0] # Take folder name in the category folder
        )
        num_rendered_models = len(render_folder_path_list)
            
        # File to collect prepared data
        hdf5_file = h5py.File(hdf5_path, 'w')
        hdf5_file.create_dataset("pixels",    [num_rendered_models, args.num_rendering, args.height, args.width, 3], np.uint8)
        if args.depth:
            hdf5_file.create_dataset("depths",    [num_rendered_models, args.num_rendering, args.height, args.width, 1], np.uint8)

        for render_indx, folder_rendered_path in tqdm(zip(range(num_rendered_models), render_folder_path_list)):
            images_list = []
            if args.depth:
                depth_list = []
            
            for i in range(args.num_rendering):
                file_image_path = os.path.join(
                    folder_rendered_path, 'rendering', 
                    f'{str(i).zfill(2)}.png'
                )
                if not os.path.isfile(file_image_path):
                    # TODO: Just skip such files?
                    raise Exception('Filename image {file_image_path} is missing.')

                images_list.append(
                    np.array(Image.open(file_image_path), dtype=np.uint8)
                )
                if args.depth:
                    file_depth_path = os.path.join(
                        folder_rendered_path, 'rendering', 
                        f'{str(i).zfill(2)}_depth_0001.png'
                    )
                    if not os.path.isfile(file_depth_path):
                        # TODO: Just skip such files?
                        raise Exception('Filename depth {file_depth_path} is missing.')
                    depth_list.append(
                        np.array(Image.open(file_depth_path), dtype=np.uint8)[:, :, 0:1]
                    )
            
            hdf5_file["pixels"][render_indx, :, :, :] = np.array(images_list, dtype=np.uint8)
            if args.depth:
                hdf5_file["depths"][render_indx, :, :, :] = np.array(depth_list, dtype=np.uint8)
        hdf5_file.close()
    print("finished")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=str,
                        help='Path to ShapeNet dataset folder where `render_blender.py` prepare views for dataset.')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save prepared data.', default='./')
    parser.add_argument('--num_rendering', 
                        help='Number of renderings view per object.', default=24)
    parser.add_argument('--width', type=int, 
                        help='Width of rendered images.', default=127)
    parser.add_argument('--height', type=int, 
                        help='Height of rendered images.', default=127)
    parser.add_argument('--depth', action='store_true',
                        help='Use generated depth per view and store to the h5 file. ')
    args = parser.parse_args()
    main(args)
    