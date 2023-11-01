import numpy as np
import os
import h5py
import argparse

import sys
sys.path.append('./../point_sample')
from constants import batch_size_1, batch_size_2, batch_size_3, vox_size_1, vox_size_2, vox_size_3, dim


def create_hdf5_file(name: str, number_elements: int, args):
    hdf5_file = h5py.File(name, 'w')
    if args.with_renders:
        hdf5_file.create_dataset("pixels",     [number_elements, args.num_rendering, args.height, args.width, len(args.image_mode)], np.uint8, compression=args.compression)
        if args.with_depth:
            hdf5_file.create_dataset("depths", [number_elements, args.num_rendering, args.height, args.width, 1], np.float32, compression=args.compression)
    
    if args.with_class_info:
        hdf5_file.create_dataset("classes", [number_elements], np.uint8, compression=args.compression)
        
    hdf5_file.create_dataset("voxels",               [number_elements, dim, dim, dim, 1], np.uint8, compression=args.compression)
    hdf5_file.create_dataset(f"points_{vox_size_1}", [number_elements, batch_size_1, 3], np.uint8, compression=args.compression)
    hdf5_file.create_dataset(f"values_{vox_size_1}", [number_elements, batch_size_1, 1], np.uint8, compression=args.compression)
    hdf5_file.create_dataset(f"points_{vox_size_2}", [number_elements, batch_size_2, 3], np.uint8, compression=args.compression)
    hdf5_file.create_dataset(f"values_{vox_size_2}", [number_elements, batch_size_2, 1], np.uint8, compression=args.compression)
    hdf5_file.create_dataset(f"points_{vox_size_3}", [number_elements, batch_size_3, 3], np.uint8, compression=args.compression)
    hdf5_file.create_dataset(f"values_{vox_size_3}", [number_elements, batch_size_3, 1], np.uint8, compression=args.compression)
    return hdf5_file


def slice_certain_ids_from_h5_to_other_h5(input_h5, output_h5, ids_input_slice_list, args):
    if args.with_renders:
        output_h5["pixels"][:] = input_h5['pixels'][ids_input_slice_list]
        if args.with_depth:
            output_h5["depths"][:] = input_h5['depths'][ids_input_slice_list]
    
    if args.with_class_info:
        output_h5["classes"][:] = input_h5['classes'][ids_input_slice_list]

    output_h5["voxels"][:] = input_h5['voxels'][ids_input_slice_list]
    output_h5[f"points_{vox_size_1}"][:] = input_h5[f"points_{vox_size_1}"][ids_input_slice_list]
    output_h5[f"values_{vox_size_1}"][:] = input_h5[f"values_{vox_size_1}"][ids_input_slice_list]
    output_h5[f"points_{vox_size_2}"][:] = input_h5[f"points_{vox_size_2}"][ids_input_slice_list]
    output_h5[f"values_{vox_size_2}"][:] = input_h5[f"values_{vox_size_2}"][ids_input_slice_list]
    output_h5[f"points_{vox_size_3}"][:] = input_h5[f"points_{vox_size_3}"][ids_input_slice_list]
    output_h5[f"values_{vox_size_3}"][:] = input_h5[f"values_{vox_size_3}"][ids_input_slice_list]
    return output_h5


def combine_and_split_h5_files(args):
    os.makedirs(args.save_folder, exist_ok=True)

    if args.compression < 0:
        args.compression = None
    elif args.compression > 9:
        args.compression = 9

    # Ids tp 
    train_ids_list = []
    test_ids_list = []
    print('Collect ids...')
    with open(args.h5_point_filename_path[:-5] + '.txt', 'r') as rf:
        all_model_ids_list = list(map(lambda x: x.strip(), rf.readlines()))
    
    if args.txt_train_ids_filename_path is not None and os.path.isfile(args.txt_train_ids_filename_path):
        with open(args.txt_train_ids_filename_path, 'r') as rf:
            train_model_ids_list = list(map(lambda x: x.strip(), rf.readlines()))
        # Make sure ids from all-models and train one are compatible
        train_model_ids_list = list(filter(lambda x: x in all_model_ids_list, train_model_ids_list))
    else:
        train_model_ids_list = []

    if args.txt_test_ids_filename_path is not None and os.path.isfile(args.txt_test_ids_filename_path):
        with open(args.txt_test_ids_filename_path, 'r') as rf:
            test_model_ids_list = list(map(lambda x: x.strip(), rf.readlines()))
        # Make sure ids from all-models and train one are compatible
        test_model_ids_list = list(filter(lambda x: x in all_model_ids_list, test_model_ids_list))
    else:
        test_model_ids_list = []

    if len(train_model_ids_list) == 0 and len(test_model_ids_list) == 0:
        num_models = len(all_model_ids_list)
        train_count = int(num_models * args.perc_train)

        all_model_ids_np = np.asarray(all_model_ids_list)
        np.random.shuffle(all_model_ids_np)
        all_model_ids_list = all_model_ids_np.tolist()
        train_model_ids_list = all_model_ids_list[:train_count]
        test_model_ids_list = all_model_ids_list[train_count:]
    
    if len(train_model_ids_list) == 0 and len(test_model_ids_list) != 0:
        # All other ids as train one
        train_model_ids_list = list(filter(lambda x: x not in test_model_ids_list, all_model_ids_list))

    if len(train_model_ids_list) != 0 and len(test_model_ids_list) == 0:
        # All other ids as test one
        test_model_ids_list = list(filter(lambda x: x not in train_model_ids_list, all_model_ids_list))

    for i, model_id in enumerate(all_model_ids_list):
        if model_id in train_model_ids_list:
            train_ids_list.append(i)
        elif model_id in test_model_ids_list:
            test_ids_list.append(i)
    print('Open and create h5 files...')
    with_all_hdf5_file = h5py.File(args.h5_point_filename_path, 'r')

    train_hdf5_file = create_hdf5_file(
        f'{args.save_folder}/dataset_train.hdf5', 
        len(train_ids_list), args
    )
    print('Write to train file...')
    train_hdf5_file = slice_certain_ids_from_h5_to_other_h5(
        with_all_hdf5_file, train_hdf5_file, 
        train_ids_list, args
    )
    # Save model ids to separate txt file
    with open(f'{args.save_folder}/dataset_train.txt', 'w') as wf:
        wf.writelines(map(lambda x: x + '\n', np.asarray(all_model_ids_list)[train_ids_list].tolist()))
        
    print('Write to test file...')
    test_hdf5_file = create_hdf5_file(
        f'{args.save_folder}/dataset_test.hdf5', 
        len(test_ids_list), args
    )

    test_hdf5_file = slice_certain_ids_from_h5_to_other_h5(
        with_all_hdf5_file, test_hdf5_file, 
        test_ids_list, args
    )
    # Save model ids to separate txt file
    with open(f'{args.save_folder}/dataset_test.txt', 'w') as wf:
        wf.writelines(map(lambda x: x + '\n', np.asarray(all_model_ids_list)[test_ids_list].tolist()))

    train_hdf5_file.close()
    test_hdf5_file.close()

    print("finished")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_point_filename_path', type=str,
                        help='Path to h5 file with data points sampled by algo.')
    parser.add_argument('--with-renders', action='store_true',
                        help='Provide this parameter to store also renders (pixels\depths). ')
    parser.add_argument('--with-class-info', action='store_true',
                        help='Provide this parameter to store also class info (plane, chair and etc...). ')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save prepared data.', default='./')
    parser.add_argument('--with-depth', action='store_true',
                        help='Use generated depth per view and store to the h5 file. ')
    parser.add_argument('--num_rendering', type=int, 
                        help='Number of renderings view per object.', default=24)
    parser.add_argument('--width', type=int, 
                        help='Width of rendered images.', default=127)
    parser.add_argument('--height', type=int, 
                        help='Height of rendered images.', default=127)
    parser.add_argument('--image-mode', choices=['L', 'RGB', 'RGBA'], 
                        help='Mode for loaded image.', default='RGBA')
    parser.add_argument('--compression', type=int, default=9,
                        help='Compression level from 0 (fast, but more memory) to 9 (slow, but low memory). '
                        'By default equal to 9. Could be negative, when no compression will be used.')
    parser.add_argument('--perc-train', type=float, default=0.8,
                        help='Percentage of training samples. By default equal to 0.8 i.e. 80% training and 20% test. ')
    parser.add_argument('--txt_train_ids_filename_path', type=str, default=None,
                        help='Path to txt file with train model ids. Other ids will be placed to test set. '
                        'NOTICE! If `txt_test_ids_filename_path` also used, then ids NOT in both txt files will be dropped.')
    parser.add_argument('--txt_test_ids_filename_path', type=str, default=None,
                        help='Path to txt file with test model ids. Other ids will be placed to train set. '
                        'NOTICE! If `txt_train_ids_filename_path` also used, then ids NOT in both txt files will be dropped.')
    args = parser.parse_args()
    combine_and_split_h5_files(args)
    