import traceback
from typing import List
import numpy as np
import os
import h5py
from multiprocessing import Process, Queue
import queue
import argparse
from dataclasses import dataclass

from tqdm import tqdm

from PIL import Image

# Add utils folder
import sys
sys.path.append('./../utils')
import binvox_rw

from utils import find_all_files_with_exts
from constants import *

from ..render_blender.constants import FOLDER_SAVE_NAME


@dataclass
class PreparedSingleData:
    indx: int
    filepath: str
    sample_voxels_1: np.ndarray
    sample_points_1: np.ndarray
    sample_values_1: np.ndarray
    
    exceed_flag_2: int
    sample_voxels_2: np.ndarray
    sample_points_2: np.ndarray
    sample_values_2: np.ndarray
    
    exceed_flag_3: int
    sample_voxels_3: np.ndarray
    sample_points_3: np.ndarray
    sample_values_3: np.ndarray

    pixels: np.ndarray = None
    depths: np.ndarray = None


@dataclass
class StoredPathData:
    binvox_path: str
    render_path: str
    is_use_depth: bool = False
    num_views: int = -1

    def add_render_path(self, render_path: str, is_use_depth: bool, num_views: int):
        self.render_path = render_path
        self.is_use_depth = is_use_depth
        self.num_views = num_views
        return self


def generate_dataset(q: Queue, args_files_path_list: List[StoredPathData], start_indx: int, end_indx: int, args):
    try:
        from point_sample_tools_numba import carve_voxels, sample_points_near_surface
    except Exception as e:
        print('Failed to run with numba, run via simple python...')
        from utils import carve_voxels, sample_points_near_surface
    for (file_path, indx) in zip(args_files_path_list, range(start_indx, end_indx)):
        try:
            # In the original repo, the author read data from .mat and prepared voxels special for ShapeNet
            #   https://github.com/czq142857/IM-NET/issues/19#issuecomment-1125435519
            # In this repo for now only .binvox supported
            # TODO: Support .mat format
            with open(file_path.binvox_path, 'rb') as f:
                voxel = binvox_rw.read_as_3d_array(f)
            voxel_model_256 = voxel.data
            # you need to make absolutely sure that the top direction of your shape is j-positive direction in the voxels,
            # otherwise the z-carving code will not work properly. (z-carving ≈ floodfill to make the voxels solid inside)
            # you can adjust the direction of your voxels via np.flip and np.transpose.
            if str(args.dataset_version) == '1':
                # add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
                voxel_model_256 = np.flip(
                    np.transpose(voxel_model_256, (2, 1, 0) ),
                    2,
                )
            carve_voxels(voxel_model_256)
            sample_voxels_3, sample_points_3, sample_values_3, exceed_flag_3 = sample_points_near_surface(
                voxel_model_256, vox_size_3, batch_size_3
            )
            sample_voxels_2, sample_points_2, sample_values_2, exceed_flag_2 = sample_points_near_surface(
                voxel_model_256, vox_size_2, batch_size_2
            )
            sample_voxels_1, sample_points_1, sample_values_1, exceed_flag_1 = sample_points_near_surface(
                voxel_model_256, vox_size_1, batch_size_1
            )

            single_data = PreparedSingleData(
                indx=indx,
                filepath=file_path.binvox_path,
                sample_voxels_1=sample_voxels_1, sample_points_1=sample_points_1, sample_values_1=sample_values_1,
                
                exceed_flag_2=exceed_flag_2,
                sample_voxels_2=sample_voxels_2, sample_points_2=sample_points_2, sample_values_2=sample_values_2,
                
                exceed_flag_3=exceed_flag_3,
                sample_voxels_3=sample_voxels_3, sample_points_3=sample_points_3, sample_values_3=sample_values_3,
            )

            # Load pixels if needed
            if file_path.render_path:
                images_list = []
                if file_path.is_use_depth:
                    depth_list = []

                for num_view_i in range(file_path.num_views):
                    file_image_path = os.path.join(
                        file_path.render_path, 
                        f'{str(num_view_i).zfill(2)}.png'
                    )
                    if not os.path.isfile(file_image_path):
                        # TODO: Just skip such files?
                        raise Exception('Filename image {file_image_path} is missing.')

                    loaded_image = Image.open(file_image_path)
                    if loaded_image.size[0] != args.width or loaded_image.size[1] != args.height:
                        loaded_image = loaded_image.resize((args.width, args.height))
                    if loaded_image.mode != args.image_mode:
                        loaded_image = loaded_image.convert(args.image_mode)
                    images_list.append(np.array(loaded_image, dtype=np.uint8))

                    if file_path.is_use_depth:
                        file_depth_path = os.path.join(
                            file_path.render_path, 
                            f'{str(num_view_i).zfill(2)}_depth_0001.png'
                        )
                        if not os.path.isfile(file_depth_path):
                            # TODO: Just skip such files?
                            raise Exception('Filename depth {file_depth_path} is missing.')
                        depth_list.append(
                            np.array(Image.open(file_depth_path), dtype=np.uint8)[:, :, 0:1]
                        )
                
                single_data.pixels = np.array(images_list, dtype=np.uint8)
                if file_path.is_use_depth:
                    single_data.depths = np.array(depth_list, dtype=np.uint8)

            q.put(single_data)
        except Exception as e:
            traceback.print_exc()
            print(f'Something goes wrong with file={file_path} and indx={indx}, skip them...')
            continue


def main(args):
    os.makedirs(args.save_folder, exist_ok=True)
    for category_with_name in class_name_list_all:
        category, name_class = category_with_name.split('_')
        print(f'Prepare dataset with category {category} and name {name_class}')
        voxel_data_path = os.path.join(args.base_dir, category)
        if not os.path.isdir(voxel_data_path):
            print(f'Part of the dataset not found!')
            continue
            
        hdf5_path = os.path.join(args.save_folder, f'{category}.hdf5')
    
        # Record statistics
        fstatistics = open(
            os.path.join(args.save_folder, f'{category}_{name_class}_statistics.txt'),
            'w', newline=''
        )
        exceed_2 = 0
        exceed_3 = 0

        binvox_files_path_list = find_all_files_with_exts(voxel_data_path, ['.binvox'])
        args_files_path_list = list(map(
            lambda path_i: StoredPathData(binvox_path=path_i, render_path=None),
            binvox_files_path_list
        ))
        if args.rendered_imgs_path:
            # For each binvox file - must be render
            args_files_path_list = map(
                lambda path_i: path_i.add_render_path(
                    os.path.join(
                        args.rendered_imgs_path, 
                        *os.path.relpath(path_i.binvox_path, args.base_dir).split(os.sep)[:2], # Take model id and category name
                        FOLDER_SAVE_NAME
                    ), args.depth, args.num_rendering,
                ),
                args_files_path_list
            )
            def is_render_exist(path_i: StoredPathData):
                if path_i.render_path is None:
                    return False
                
                # Check if render exist
                if not os.path.isfile(os.path.join(path_i.render_path, f'{str(args.num_rendering-1).zfill(2)}.png')):
                    return False
                
                if args.depth and not os.path.isfile(
                        os.path.join(path_i.render_path, f'{str(args.num_rendering-1).zfill(2)}_depth_0001.png')):
                    return False

                return True
            # Filter files that does not have renderings - just skip them
            args_files_path_list = list(filter(
                is_render_exist,
                args_files_path_list
            ))
        args_files_path_list = sorted(
            args_files_path_list,
            key=lambda x: os.path.relpath(x.binvox_path, voxel_data_path).split(os.sep)[0] # Take folder name in the category folder
        )
        num_of_binvox_files = len(args_files_path_list)

        # Write category and model id paths to separate file
        idx2model_id_txt_path = os.path.join(args.save_folder, f'{category}.txt')
        with open(idx2model_id_txt_path,'w',newline='') as fout:
            for single_path_data in args_files_path_list:
                fout.write(os.sep.join(os.path.relpath(single_path_data.binvox_path, args.base_dir).split(os.sep)[:2])+"\n")
        
        # Make list for each process
        number_files_per_process = num_of_binvox_files // args.num_process
        args_to_generate_dataset_per_process = [
            (
                args_files_path_list[i * number_files_per_process: (i+1) * number_files_per_process],
                i * number_files_per_process, (i+1) * number_files_per_process
            )
            for i in range(args.num_process-1)
        ]
        # Append the remaining files for last process
        last_indx_start = (args.num_process-1) * number_files_per_process
        last_files = args_files_path_list[last_indx_start:]
        args_to_generate_dataset_per_process.append(
            (
                last_files,
                last_indx_start, 
                last_indx_start + len(last_files)
            )
        )
        
        q = Queue()
        workers = [
            Process(target=generate_dataset, args = (q, args_files_path_list, start_indx, end_indx, args)) 
            for args_files_path_list, start_indx, end_indx in args_to_generate_dataset_per_process
        ]

        for p in workers:
            p.start()
            
        # File to collect prepared data
        hdf5_file = h5py.File(hdf5_path, 'w')
        # TODO: Add more voxels? Here its with size 64
        hdf5_file.create_dataset("voxels", [num_of_binvox_files, dim, dim, dim, 1], np.uint8)
        # Do not add compression parameter otherwise this loop will be very slow
        # Compression will be done when all files (h5) will be combined into single one
        hdf5_file.create_dataset(f"points_{vox_size_1}", [num_of_binvox_files, batch_size_1, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_1}", [num_of_binvox_files, batch_size_1, 1], np.uint8)
        hdf5_file.create_dataset(f"points_{vox_size_2}", [num_of_binvox_files, batch_size_2, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_2}", [num_of_binvox_files, batch_size_2, 1], np.uint8)
        hdf5_file.create_dataset(f"points_{vox_size_3}", [num_of_binvox_files, batch_size_3, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_3}", [num_of_binvox_files, batch_size_3, 1], np.uint8)

        # Additional data
        if args.rendered_imgs_path:
            hdf5_file.create_dataset(f"pixels", [num_of_binvox_files, args.num_rendering, args.height, args.width, len(args.image_mode)], np.uint8)
            if args.depth:
                hdf5_file.create_dataset(f"depths", [num_of_binvox_files, args.num_rendering, args.height, args.width, 1], np.uint8)

        with tqdm(total=num_of_binvox_files) as pbar:
            while True:
                item_flag = True
                try:
                    single_data: PreparedSingleData = q.get(True, 1.0)
                except queue.Empty:
                    item_flag = False

                if item_flag:
                    exceed_2 += single_data.exceed_flag_2
                    exceed_3 += single_data.exceed_flag_3
                    hdf5_file[f"points_{vox_size_1}"][single_data.indx,:,:] = single_data.sample_points_1
                    hdf5_file[f"values_{vox_size_1}"][single_data.indx,:,:] = single_data.sample_values_1
                    hdf5_file[f"points_{vox_size_2}"][single_data.indx,:,:] = single_data.sample_points_2
                    hdf5_file[f"values_{vox_size_2}"][single_data.indx,:,:] = single_data.sample_values_2
                    hdf5_file[f"points_{vox_size_3}"][single_data.indx,:,:] = single_data.sample_points_3
                    hdf5_file[f"values_{vox_size_3}"][single_data.indx,:,:] = single_data.sample_values_3
                    hdf5_file["voxels"][single_data.indx,:,:,:,:] = single_data.sample_voxels_3

                    # Additional data
                    if args.rendered_imgs_path:
                        hdf5_file["pixels"][single_data.indx,:,:,:,:] = single_data.pixels
                        if args.depth:
                            hdf5_file["depths"][single_data.indx,:,:,:,:] = single_data.depths
                    pbar.update()

                is_any_process_alive = False
                for p in workers:
                    if p.exitcode is None:
                        is_any_process_alive = True
                        break
                if not is_any_process_alive and q.empty():
                    break
        fstatistics.write(f"total: {num_of_binvox_files}\n")
        fstatistics.write(f"exceed_2: {exceed_2}\n"+str(exceed_2)+"\n")
        fstatistics.write(f"exceed_2_ratio: {float(exceed_2)/num_of_binvox_files}\n")
        fstatistics.write(f"exceed_3: {exceed_3}\n")
        fstatistics.write(f"exceed_3_ratio: {float(exceed_3)/num_of_binvox_files}\n")
        
        fstatistics.close()
        hdf5_file.close()
    print("finished")

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=str,
                        help='Path to ShapeNet dataset folder with binvox files.')
    parser.add_argument('-r', '--rendered_imgs_path', type=str,
                        help='Path to folder with rendered images by Blender.')
    parser.add_argument('--num_rendering', type=int, 
                        help='Number of renderings view per object.', default=24)
    parser.add_argument('--depth', action='store_true',
                        help='Store depth inside h5 file, otherwise only pixels are saved. ')
    parser.add_argument('--width', type=int, 
                        help='Width of rendered images.', default=127)
    parser.add_argument('--height', type=int, 
                        help='Height of rendered images.', default=127)
    parser.add_argument('--image-mode', choices=['L', 'RGB', 'RGBA'], 
                        help='Mode for loaded image.', default='RGBA')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save prepared data.', default='./')
    parser.add_argument('-d', '--dataset-version', choices=['1', '2'], 
                        help='Choose version of ShapeNet dataset. If version is 1, '
                        'when binvox coordinates will be mapped to coordinates for version 2', default='2')
    parser.add_argument('-n', '--num-process', type=int, 
                        help='Number of process.', default=4)
    args = parser.parse_args()
    main(args)
    