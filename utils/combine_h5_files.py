import traceback
import numpy as np
import os
import h5py
import argparse

from tqdm import tqdm
import glob

import sys
sys.path.append('./../point_sample')
from constants import batch_size_1, batch_size_2, batch_size_3, vox_size_1, vox_size_2, vox_size_3, dim


class H5FilesController:

    def __init__(
            self, number_elements, args, 
            num_trains=None, num_tests=None):
        self.args = args

        self.hdf5_file = None

        self.hdf5_train_file = None
        self.hdf5_test_file = None

        if not args.split:
            print(f'Size {number_elements}')
            self.hdf5_file, self.indx2model_id_list = self._create_hdf5_file(
                f'{args.save_folder}/dataset.hdf5', number_elements, args
            )

            self.txt_indx2model_id_path = f'{args.save_folder}/dataset.txt'

            self.split_mode = False
        else:
            assert num_trains is not None and num_tests is not None
            assert number_elements == (num_trains + num_tests)
            print(f'Size train={num_trains}, test={num_tests}')
            self.hdf5_train_file, self.train_indx2model_id_list = self._create_hdf5_file(
                f'{args.save_folder}/dataset_train.hdf5', 
                num_trains, args
            )
            self.hdf5_test_file, self.test_indx2model_id_list = self._create_hdf5_file(
                f'{args.save_folder}/dataset_test.hdf5', 
                num_tests, args
            )

            self.train_txt_indx2model_id_path = f'{args.save_folder}/dataset_train.txt'
            self.test_txt_indx2model_id_path = f'{args.save_folder}/dataset_test.txt'

            self.split_mode = True
    
    def _create_hdf5_file(self, name: str, number_elements: int, args):
        hdf5_file = h5py.File(name, 'w')
        if args.with_renders:
            hdf5_file.create_dataset("pixels",     [number_elements, args.num_rendering, args.height, args.width, len(args.image_mode)], np.uint8, compression=args.compression)
            if args.depth:
                hdf5_file.create_dataset("depths", [number_elements, args.num_rendering, args.height, args.width, 1], np.float32, compression=args.compression)
        
        if args.add_class_info:
            hdf5_file.create_dataset("classes", [number_elements], np.uint8, compression=args.compression)
            
        hdf5_file.create_dataset("voxels",               [number_elements, dim, dim, dim, 1], np.uint8, compression=args.compression)
        hdf5_file.create_dataset(f"points_{vox_size_1}", [number_elements, batch_size_1, 3], np.uint8, compression=args.compression)
        hdf5_file.create_dataset(f"values_{vox_size_1}", [number_elements, batch_size_1, 1], np.uint8, compression=args.compression)
        hdf5_file.create_dataset(f"points_{vox_size_2}", [number_elements, batch_size_2, 3], np.uint8, compression=args.compression)
        hdf5_file.create_dataset(f"values_{vox_size_2}", [number_elements, batch_size_2, 1], np.uint8, compression=args.compression)
        hdf5_file.create_dataset(f"points_{vox_size_3}", [number_elements, batch_size_3, 3], np.uint8, compression=args.compression)
        hdf5_file.create_dataset(f"values_{vox_size_3}", [number_elements, batch_size_3, 1], np.uint8, compression=args.compression)

        return hdf5_file, [None] * number_elements if args.create_indx2model_id_file else None


    def _write_to_h5_file(
            self, hdf5_file_to: h5py.File, hdf5_file_from: h5py.File, 
            start_to: int=None, end_to: int=None, 
            start_from: int=None, end_from: int=None, class_number: int=None):
        if start_to is not None and end_to is not None and start_from is None and end_from is None:
            hdf5_file_to[f"points_{vox_size_1}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_1}"][:]
            hdf5_file_to[f"values_{vox_size_1}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_1}"][:]
            hdf5_file_to[f"points_{vox_size_2}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_2}"][:]
            hdf5_file_to[f"values_{vox_size_2}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_2}"][:]
            hdf5_file_to[f"points_{vox_size_3}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_3}"][:]
            hdf5_file_to[f"values_{vox_size_3}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_3}"][:]
            hdf5_file_to["voxels"][start_to: end_to] = hdf5_file_from["voxels"][:]

            if self.args.with_renders:
                hdf5_file_to["pixels"][start_to: end_to] = hdf5_file_from["pixels"][:]
                if self.args.depth:
                    hdf5_file_to["depths"][start_to: end_to] = hdf5_file_from["depths"][:]
            
            if self.args.add_class_info:
                if class_number is None:
                    raise Exception('Error! `add_class_info` is True, but class_number in write is None.')
                hdf5_file_to["classes"][start_to: end_to] = np.array([class_number] * (end_to - start_to), dtype=np.uint8)
        elif start_to is not None and end_to is not None and start_from is not None and end_from is not None:
            hdf5_file_to[f"points_{vox_size_1}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_1}"][start_from: end_from]
            hdf5_file_to[f"values_{vox_size_1}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_1}"][start_from: end_from]
            hdf5_file_to[f"points_{vox_size_2}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_2}"][start_from: end_from]
            hdf5_file_to[f"values_{vox_size_2}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_2}"][start_from: end_from]
            hdf5_file_to[f"points_{vox_size_3}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_3}"][start_from: end_from]
            hdf5_file_to[f"values_{vox_size_3}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_3}"][start_from: end_from]
            hdf5_file_to["voxels"][start_to: end_to] = hdf5_file_from["voxels"][start_from: end_from]

            if self.args.with_renders:
                hdf5_file_to["pixels"][start_to: end_to] = hdf5_file_from["pixels"][start_from: end_from]
                if self.args.depth:
                    hdf5_file_to["depths"][start_to: end_to] = hdf5_file_from["depths"][start_from: end_from]
            
            if self.args.add_class_info:
                if class_number is None:
                    raise Exception('Error! `add_class_info` is True, but class_number in write is None.')
                hdf5_file_to["classes"][start_to: end_to] = np.array([class_number] * (end_to - start_to), dtype=np.uint8)
        elif start_to is not None and end_to is not None and start_from is not None and end_from is None:
            hdf5_file_to[f"points_{vox_size_1}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_1}"][start_from:]
            hdf5_file_to[f"values_{vox_size_1}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_1}"][start_from:]
            hdf5_file_to[f"points_{vox_size_2}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_2}"][start_from:]
            hdf5_file_to[f"values_{vox_size_2}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_2}"][start_from:]
            hdf5_file_to[f"points_{vox_size_3}"][start_to: end_to] = hdf5_file_from[f"points_{vox_size_3}"][start_from:]
            hdf5_file_to[f"values_{vox_size_3}"][start_to: end_to] = hdf5_file_from[f"values_{vox_size_3}"][start_from:]
            hdf5_file_to["voxels"][start_to: end_to] = hdf5_file_from["voxels"][start_from:]

            if self.args.with_renders:
                hdf5_file_to["pixels"][start_to: end_to] = hdf5_file_from["pixels"][start_from:]
                if self.args.depth:
                    hdf5_file_to["depths"][start_to: end_to] = hdf5_file_from["depths"][start_from:]
            
            if self.args.add_class_info:
                if class_number is None:
                    raise Exception('Error! `add_class_info` is True, but class_number in write is None.')
                hdf5_file_to["classes"][start_to: end_to] = np.array([class_number] * (end_to - start_to), dtype=np.uint8)
        else:
            raise Exception('Not supported values of parameters. ')

    def _append_indx2model_id(self, indx2model_id_to_list: list, indx2model_id_to_from: list, 
            start_to: int=None, end_to: int=None, 
            start_from: int=None, end_from: int=None):
        
        if start_to is not None and end_to is not None and start_from is None and end_from is None:
            indx2model_id_to_list[start_to: end_to] = indx2model_id_to_from[:]
        elif start_to is not None and end_to is not None and start_from is not None and end_from is not None:
            indx2model_id_to_list[start_to: end_to] = indx2model_id_to_from[start_from: end_from]
        elif start_to is not None and end_to is not None and start_from is not None and end_from is None:
            indx2model_id_to_list[start_to: end_to] = indx2model_id_to_from[start_from:]
        else:
            raise Exception('Not supported values of parameters. ')

    def _write_indx2model_id_to_file(self, filepath: str, indx2model_id_list: list):
        with open(filepath,'w',newline='') as fout:
            for i in range(len(indx2model_id_list)):
                fout.write(indx2model_id_list[i] + '\n')
    
    def write(self, hdf5_file: h5py.File, start, end, indx2model_id_list: list=None, class_number: int=None):
        if self.split_mode:
            raise Exception('Write without split is called but current mode is for split. ')
        self._write_to_h5_file(self.hdf5_file, hdf5_file, start, end, class_number=class_number)
        if indx2model_id_list is not None:
            self._append_indx2model_id(self.indx2model_id_list, indx2model_id_list, start, end)

    def write_split(
            self, hdf5_file: h5py.File, 
            num_train, start_train, end_train, 
            start_test, end_test, 
            indx2model_id_list: list=None, class_number: int=None):
        if not self.split_mode:
            raise Exception('Write with split is called but current mode is for not split. ')
        self._write_to_h5_file(self.hdf5_train_file, hdf5_file, start_train, end_train, 0, num_train, class_number=class_number)
        self._write_to_h5_file(self.hdf5_test_file, hdf5_file, start_test, end_test, num_train, class_number=class_number)

        if indx2model_id_list is not None:
            self._append_indx2model_id(self.train_indx2model_id_list, indx2model_id_list, start_train, end_train, 0, num_train)
            self._append_indx2model_id(self.test_indx2model_id_list, indx2model_id_list, start_test, end_test, num_train)

    def close(self):
        if self.split_mode:
            self.hdf5_train_file.close()
            self.hdf5_test_file.close()
            if self.train_indx2model_id_list is not None:
                if None in self.train_indx2model_id_list:
                    raise Exception('Error! In `train_indx2model_id_list` was found None value for id')
                self._write_indx2model_id_to_file(self.train_txt_indx2model_id_path, self.train_indx2model_id_list)

            if self.test_indx2model_id_list is not None:
                if None in self.test_indx2model_id_list:
                    raise Exception('Error! In `test_indx2model_id_list` was found None value for id')
                self._write_indx2model_id_to_file(self.test_txt_indx2model_id_path, self.test_indx2model_id_list)
        else:
            self.hdf5_file.close()

            if self.indx2model_id_list is not None:
                if None in self.indx2model_id_list:
                    raise Exception('Error! In `indx2model_id_list` was found None value for id')
                self._write_indx2model_id_to_file(self.txt_indx2model_id_path, self.indx2model_id_list)


def combine_and_split_h5_files(args):
    os.makedirs(args.save_folder, exist_ok=True)

    if args.compression < 0:
        print('No compression.')
        args.compression = None
    elif args.compression > 9:
        print('Compression bigger than 9, set to 9.')
        args.compression = 9
    else:
        print(f'Compression level {args.compression}')

    number_elements = 0
    number_train_elements = 0
    number_test_elements = 0
    start = 0
    start_train = 0
    start_test = 0

    h5_filename_list = []
    h5_point_files = []
    h5_file_slice_info_list = []

    for h5_point_file_path in glob.glob(f'{args.h5_point_files_path}/*.hdf5'):
        hdf5_point_file = h5py.File(h5_point_file_path, 'r')
        _, filename = os.path.split(h5_point_file_path)
        filename = filename.split('.')[0] # Remove type

        num_elements_point = hdf5_point_file['voxels'].shape[0]
        number_elements += num_elements_point
        end = start + num_elements_point

        if not args.split:
            h5_file_slice_info_list.append((start, end))
        else:
            # train | test
            num_train = int(num_elements_point * args.perc_train)
            num_test = num_elements_point - num_train
            end_train = start_train + num_train
            end_test = start_test + num_test
            h5_file_slice_info_list.append((
                (num_train), 
                (start_train, end_train), 
                (start_test, end_test)
            ))

            number_train_elements += num_train
            number_test_elements += num_test

            start_train = end_train
            start_test = end_test
        h5_point_files.append(hdf5_point_file)
        h5_filename_list.append(filename)

        start = end
    class_names_list = list(sorted(list(set(h5_filename_list))))
    hdf5_file_controller = H5FilesController(
        number_elements, args, 
        num_trains=number_train_elements, num_tests=number_test_elements,
    )
    try:
        for (filename_s, hdf5_point_file, slice_info)  in tqdm(zip(
                h5_filename_list, h5_point_files, h5_file_slice_info_list) ):
            class_name = class_names_list.index(filename_s) if args.add_class_info else None

            indx2model_id_list = None
            if args.create_indx2model_id_file:
                with open(f'{args.h5_point_files_path}/{filename_s}.txt', 'r') as fr:
                    indx2model_id_list = list(map(lambda x: x.strip(), fr.readlines()))

            if not args.split:
                start, end = slice_info
                hdf5_file_controller.write(
                    hdf5_point_file, start, end, 
                    indx2model_id_list=indx2model_id_list,
                    class_number=class_name,
                )
            else:
                (num_train), (start_train, end_train), (start_test, end_test) = slice_info
                hdf5_file_controller.write_split(
                    hdf5_point_file, num_train, 
                    start_train, end_train, 
                    start_test, end_test, 
                    indx2model_id_list=indx2model_id_list,
                    class_number=class_name,
                )
            hdf5_point_file.close()
    except Exception as e:
        traceback.print_exc()
    finally:
        hdf5_file_controller.close()
    print("finished")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_point_files_path', type=str,
                        help='Path to folder with h5 files by point sample algo.')
    parser.add_argument('--with-renders', action='store_true',
                        help='Provide this parameter to store also renders (pixels\depths). ')
    parser.add_argument('--add-class-info', action='store_true',
                        help='If provided, when `classes` parameter will be created where single class corresponding for single h5 file.'
                        'Indexes assigned as sorted string. ')
    parser.add_argument('--create-indx2model-id-file', action='store_true',
                        help='Generated h5 files from point sample also comes with txt files '
                        'where each string correspond to index in h5 file. ')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save prepared data.', default='./')
    parser.add_argument('--depth', action='store_true',
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
    parser.add_argument('--split', action='store_true',
                        help='To split to train and test files provide this parameter. '
                        'By default all data will be saved in single file. '
                        'More flexable split presented in `split_h5_file_to_train_test.py` script. ')
    parser.add_argument('--perc-train', type=float, default=0.8,
                        help='Percentage of training samples. By default equal to 0.8 i.e. 80% training and 20% test. '
                        'More flexable split presented in `split_h5_file_to_train_test.py` script. ')
    args = parser.parse_args()
    combine_and_split_h5_files(args)
    