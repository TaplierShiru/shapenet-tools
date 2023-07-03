import traceback
import numpy as np
import os
import h5py
import argparse


def add_class_info_to_h5_file(args):
    hdf5_file = h5py.File(args.h5_file_path, 'a')
    try:
        obj_paths = [line.rstrip('\n') for line in open(args.txt_file_path, mode='r').readlines()]
        num_elements = hdf5_file['voxels'].shape[0]
        assert num_elements == len(obj_paths)

        classes_names = list(set(map(lambda x: x.split(os.sep)[0], obj_paths)))
        classes_names = list(sorted(classes_names))
        indx2class = map(lambda x: classes_names.index(x.split(os.sep)[0]), obj_paths)
        indx2class = np.array(list(indx2class), dtype=np.uint8).reshape((-1,))

        hdf5_file.create_dataset("classes", [num_elements], np.uint8, compression=9)
        hdf5_file['classes'][:] = indx2class[:]

    except Exception as e:
        traceback.print_exc()
    finally:
        hdf5_file.close()
    print("finished")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_file_path', type=str,
                        help='Path to h5 file in which you need to append class info.')
    parser.add_argument('txt_file_path', type=str,
                        help='Path to txt file with information about category and model id for each index in h5 file.')
    args = parser.parse_args()
    add_class_info_to_h5_file(args)
    