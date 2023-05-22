import numpy as np
import os
import h5py
from multiprocessing import Process, Queue
import queue
import time
import argparse
from dataclasses import dataclass

from tqdm import tqdm

# Add utils folder
import sys
sys.path.append('../utils')
import binvox_rw

from point_sample_tools import find_all_files_with_exts

class_name_list_all = [
    "02691156_airplane",
    "02828884_bench",
    "02933112_cabinet",
    "02958343_car",
    "03001627_chair",
    "03211117_display",
    "03636649_lamp",
    "03691459_speaker",
    "04090263_rifle",
    "04256520_couch",
    "04379243_table",
    "04401088_phone",
    "04530566_vessel",
]

dim = 64

dataset_version = 2

vox_size_1 = 16
vox_size_2 = 32
vox_size_3 = 64

batch_size_1 = 16*16*16
batch_size_2 = 16*16*16
batch_size_3 = 16*16*16*4


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


def generate_dataset(q: Queue, binvox_files_path_list, start_indx: int, end_indx: int):
    try:
        from point_sample_tools_numba import carve_voxels, sample_points_near_surface
    except Exception as e:
        print('Failed to run with numba, run via simple python...')
        from point_sample_tools import carve_voxels, sample_points_near_surface
    for (binvox_file_path, indx) in zip(binvox_files_path_list, range(start_indx, end_indx)):
        # In the original repo, the author read data from .mat and prepared voxels special for ShapeNet
        #   https://github.com/czq142857/IM-NET/issues/19#issuecomment-1125435519
        # In this repo for now only .binvox supported
        # TODO: Support .mat format
        with open(binvox_file_path, 'rb') as f:
            voxel = binvox_rw.read_as_3d_array(f)
        voxel_model_256 = voxel.data
        # you need to make absolutely sure that the top direction of your shape is j-positive direction in the voxels,
        # otherwise the z-carving code will not work properly. (z-carving ≈ floodfill to make the voxels solid inside)
        # you can adjust the direction of your voxels via np.flip and np.transpose.
        if dataset_version == 1:
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
            filepath=binvox_file_path,
            sample_voxels_1=sample_voxels_1, sample_points_1=sample_points_1, sample_values_1=sample_values_1,
            
            exceed_flag_2=exceed_flag_2,
            sample_voxels_2=sample_voxels_2, sample_points_2=sample_points_2, sample_values_2=sample_values_2,
            
            exceed_flag_3=exceed_flag_3,
            sample_voxels_3=sample_voxels_3, sample_points_3=sample_points_3, sample_values_3=sample_values_3,
        )
        q.put(single_data)


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
        exceed_32 = 0
        exceed_64 = 0

        binvox_files_path_list = find_all_files_with_exts(voxel_data_path, ['.binvox'])
        num_of_binvox_files = len(binvox_files_path_list)
        
        # Make list for each process
        number_files_per_process = num_of_binvox_files // args.num_process
        args_to_generate_dataset_per_process = [
            (
                binvox_files_path_list[i * number_files_per_process: (i+1) * number_files_per_process],
                i * number_files_per_process, (i+1) * number_files_per_process
            )
            for i in range(args.num_process-1)
        ]
        # Append the remaining files for last process
        last_indx_start = (args.num_process-1) * number_files_per_process
        last_files = binvox_files_path_list[last_indx_start:]
        args_to_generate_dataset_per_process.append(
            (
                last_files,
                last_indx_start, 
                last_indx_start + len(last_files)
            )
        )
        
        q = Queue()
        workers = [
            Process(target=generate_dataset, args = (q, binvox_files_path_list, start_indx, end_indx)) 
            for binvox_files_path_list, start_indx, end_indx in args_to_generate_dataset_per_process
        ]

        for p in workers:
            p.start()
            
        # File to collect prepared data
        hdf5_file = h5py.File(hdf5_path, 'w')
        # TODO: Add more voxels? Here its with size 64
        hdf5_file.create_dataset("voxels",    [num_of_binvox_files, dim, dim, dim, 1], np.uint8)

        hdf5_file.create_dataset(f"points_{vox_size_1}", [num_of_binvox_files, batch_size_1, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_1}", [num_of_binvox_files, batch_size_1, 1], np.uint8)
        hdf5_file.create_dataset(f"points_{vox_size_2}", [num_of_binvox_files, batch_size_2, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_2}", [num_of_binvox_files, batch_size_2, 1], np.uint8)
        hdf5_file.create_dataset(f"points_{vox_size_3}", [num_of_binvox_files, batch_size_3, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_3}", [num_of_binvox_files, batch_size_3, 1], np.uint8)

        with tqdm(total=num_of_binvox_files) as pbar:
            while True:
                item_flag = True
                try:
                    single_data: PreparedSingleData = q.get(True, 1.0)
                except queue.Empty:
                    item_flag = False

                if item_flag:
                    exceed_32 += single_data.exceed_flag_2
                    exceed_64 += single_data.exceed_flag_3
                    hdf5_file["points_64"][single_data.indx,:,:] = single_data.sample_points_3
                    hdf5_file["values_64"][single_data.indx,:,:] = single_data.sample_values_3
                    hdf5_file["points_32"][single_data.indx,:,:] = single_data.sample_points_2
                    hdf5_file["values_32"][single_data.indx,:,:] = single_data.sample_values_2
                    hdf5_file["points_16"][single_data.indx,:,:] = single_data.sample_points_1
                    hdf5_file["values_16"][single_data.indx,:,:] = single_data.sample_values_1
                    hdf5_file["voxels"][single_data.indx,:,:,:,:] = single_data.sample_voxels_3

                is_any_process_alive = False
                for p in workers:
                    if p.exitcode is None:
                        is_any_process_alive = True
                        break
                if not is_any_process_alive and q.empty():
                    break
                pbar.update()
        fstatistics.write(f"total: {num_of_binvox_files}\n")
        fstatistics.write(f"exceed_32: {exceed_32}\n"+str(exceed_32)+"\n")
        fstatistics.write(f"exceed_32_ratio: {float(exceed_32)/num_of_binvox_files}\n")
        fstatistics.write(f"exceed_64: {exceed_64}\n")
        fstatistics.write(f"exceed_64_ratio: {float(exceed_64)/num_of_binvox_files}\n")
        
        fstatistics.close()
        hdf5_file.close()
    print("finished")

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=str,
                        help='Path to ShapeNet dataset folder with binvox files.')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save prepared data.', default='./')
    parser.add_argument('-d', '--dataset-version', choices=[1, 2], 
                        help='Choose version of ShapeNet dataset.', default=2)
    parser.add_argument('-n', '--num-process', type=int, 
                        help='Number of process.', default=4)
    args = parser.parse_args()
    main(args)
    