import traceback
import numpy as np
import os
import h5py
import argparse

from tqdm import tqdm
import glob

from ..point_sample.constants import batch_size_1, batch_size_2, batch_size_3, vox_size_1, vox_size_2, vox_size_3, dim


def main(args):
    if args.h5_render_files_path:
        print(
            'Notice! Its better to concat with render files via point_sample script. '
            'Just provide additional path to the rendered images to add them into final h5 file. '
            'After that you can just concat generated h5 files.'
        )
    os.makedirs(args.save_folder, exist_ok=True)
    number_elements = 0
    h5_filename_list = []
    h5_point_files = []
    if args.h5_render_files_path:
        h5_render_files = []
        assert len(glob.glob(f'{args.h5_point_files_path}/*.hdf5')) == len(glob.glob(f'{args.h5_render_files_path}/*.hdf5'))

    for h5_point_file_path in glob.glob(f'{args.h5_point_files_path}/*.hdf5'):
        hdf5_point_file = h5py.File(h5_point_file_path, 'r')
        filename = h5_point_file_path.split('.')[0].split('_')[0]
        num_elements_point = hdf5_point_file['voxels'].shape[0]
        number_elements += num_elements_point
        h5_point_files.append(hdf5_point_file)

        if args.h5_render_files_path:
            h5_render_file_path = os.path.join(args.h5_render_files_path, f'{filename}_render.hdf5')
            hdf5_render_file = h5py.File(h5_render_file_path, 'r')
            num_elements_render = hdf5_point_file['pixels'].shape[0]
            if num_elements_point != num_elements_render:
                raise Exception(f'For {filename} number of point elements and render not equal.')
            h5_render_files.append(hdf5_render_file)
        h5_filename_list.append(filename)

    hdf5_file = h5py.File(f'{args.save_folder}/dataset.hdf5', 'w')
    try:
        if args.h5_render_files_path:
            hdf5_file.create_dataset("pixels", [number_elements, args.num_rendering, args.height, args.width, 3], np.uint8)
            if args.depth:
                hdf5_file.create_dataset("depths", [number_elements, args.num_rendering, args.height, args.width, 1], np.uint8)
            
        hdf5_file.create_dataset("voxels", [number_elements, dim, dim, dim, 1], np.uint8)
        hdf5_file.create_dataset(f"points_{vox_size_1}", [number_elements, batch_size_1, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_1}", [number_elements, batch_size_1, 1], np.uint8)
        hdf5_file.create_dataset(f"points_{vox_size_2}", [number_elements, batch_size_2, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_2}", [number_elements, batch_size_2, 1], np.uint8)
        hdf5_file.create_dataset(f"points_{vox_size_3}", [number_elements, batch_size_3, 3], np.uint8)
        hdf5_file.create_dataset(f"values_{vox_size_3}", [number_elements, batch_size_3, 1], np.uint8)

        start = 0
        for indx_file in tqdm(range(len(h5_point_files))):
            hdf5_point_file = h5_point_files[indx_file]
            size = hdf5_point_file['voxels'].shape[0]
            end = start + size 

            hdf5_file[f"points_{vox_size_1}"][start: end, :, :] = hdf5_point_file[f"points_{vox_size_1}"][:]
            hdf5_file[f"values_{vox_size_1}"][start: end, :, :] = hdf5_point_file[f"values_{vox_size_1}"][:]
            hdf5_file[f"points_{vox_size_2}"][start: end, :, :] = hdf5_point_file[f"points_{vox_size_2}"][:]
            hdf5_file[f"values_{vox_size_2}"][start: end, :, :] = hdf5_point_file[f"values_{vox_size_2}"][:]
            hdf5_file[f"points_{vox_size_3}"][start: end, :, :] = hdf5_point_file[f"points_{vox_size_3}"][:]
            hdf5_file[f"values_{vox_size_3}"][start: end, :, :] = hdf5_point_file[f"values_{vox_size_3}"][:]
            hdf5_file["voxels"][start: end, :, :, :] = hdf5_point_file["voxels"][:]
            hdf5_point_file.close()

            if args.h5_render_files_path:
                hdf5_render_file = h5_render_files[indx_file]
                size_render = hdf5_render_file["pixels"].shape[0]
                if size_render != size:
                    raise Exception(
                        f'Size of render h5 file {h5_filename_list[indx_file]} '
                        f'have {size_render} records, but points have {size} records. '
                    )
                hdf5_file["pixels"][start: end, :, :, :] = hdf5_render_file["pixels"][:]
                if args.depth:
                    hdf5_file["depths"][start: end, :, :, :] = hdf5_render_file["depths"][:]
                hdf5_render_file.close()
            start = end
    except Exception as e:
        traceback.print_exc()
    finally:
        hdf5_file.close()
    print("finished")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_point_files_path', type=str,
                        help='Path to folder with h5 files by point sample algo.')
    parser.add_argument('--h5_render_files_path', type=str,
                        help='Path to folder with h5 files by render. If None (default) when only point files will be combined '
                        'otherwise combine pixels and point data into single h5 file', default=None)
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save prepared data.', default='./')
    parser.add_argument('--depth', action='store_true',
                        help='Use generated depth per view and store to the h5 file. ')
    args = parser.parse_args()
    main(args)
    