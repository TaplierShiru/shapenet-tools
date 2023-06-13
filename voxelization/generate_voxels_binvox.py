import os
import argparse
from multiprocessing import Process
import subprocess


CONST_R2N2_DATASET = [
    '02691156', '02933112', '03001627', '03636649', 
    '04090263', '04379243', '04530566', '02828884', 
    '02958343', '03211117', '03691459', '04256520', 
    '04401088'  
]

def generate_voxel(
        filename, file_save_path, voxel_size, 
        use_virtal_screen=False, is_in_unit_cube=True, is_exact_voxel=True):
    binvox_command = './binvox'

    if is_in_unit_cube:
        # Needed for generation of the proper voxels
        # The main purpose of this box is not fix size of all objects to around same size
        # And also center them
        # See readme for better understanding
        binvox_command = f'{binvox_command} -bb -0.5 -0.5 -0.5 0.5 0.5 0.5'
    
    if is_exact_voxel:
        binvox_command = f'{binvox_command} -e'

    if use_virtal_screen:
        binvox_command = f'Xvfb :1 -screen 0 1900x1080x24+32 & export DISPLAY=:1 && {binvox_command}'

    binvox_command = f'{binvox_command} -d {voxel_size} {filename}'

    # Generate voxel
    subprocess.run(binvox_command, shell=True)
    # Move generated voxel to save folder
    dir_to_save, filename_to_save = os.path.split(file_save_path)
    dir_origin, filename_origin = os.path.split(filename)
    subprocess.run(
        f'mkdir -p {dir_to_save}; mv {os.path.join(dir_origin, filename_to_save)} {dir_to_save}', 
        shell=True
    )

    
def start_generate_voxels(id_process, args_batch):
    for args in args_batch:
        generate_voxel(*args)
        
    
def main(args):
    print(f'Gather model files from: {args.base_dir}')
    
    args_to_generate_voxels = []
    for dirpath, dirnames, filenames in os.walk(args.base_dir):
        # Skip certain folders
        if args.datatset_type == 'r2n2' and not any(map(lambda x: x in dirpath, CONST_R2N2_DATASET)):
            continue
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in [f'.{args.type_model}']):
                args_to_generate_voxels.append((
                    fpath,
                    os.path.join(
                        dirpath.replace(args.base_dir, args.save_folder),
                        fname.replace(f'.{args.type_model}', '.binvox'),
                    ),
                    args.voxel_size,
                    args.virtual_display, args.in_unit_cube, args.exact_generation
                ))
    print(f'Found models: {len(args_to_generate_voxels)}')
    # Make list for each process
    number_files_per_process = len(args_to_generate_voxels) // args.num_process
    args_to_generate_voxels_per_process = [
        args_to_generate_voxels[i * number_files_per_process: (i+1) * number_files_per_process]
        for i in range(args.num_process-1)
    ]
    # Append the remaining files for last process
    args_to_generate_voxels_per_process.append(
        args_to_generate_voxels[args.num_process-1:]
    )
    
    workers = [
        Process(
            target=start_generate_voxels, 
            args = (i+1, args_to_generate_voxels_per_process[i])
        ) 
        for i in range(args.num_process)
    ]
    
    try:
        for p in workers:
            p.start()

        for p in workers:
            p.join()
    except Exception as e:
        print('Close processes...')
        for p in workers:
            p.kill()
    

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=str,
                        help='Path to ShapeNet dataset folder with obj files.')
    parser.add_argument('-v', '--voxel-size', type=int,
                        help='Size of the single side for voxel. Default equal to 256',
                        default=256)
    parser.add_argument('-t', '--type-model', type=str,
                        help='Type of 3d model which will be voxelized. Default equal to obj',
                        default='obj')
    parser.add_argument('-d', '--datatset-type', choices=['r2n2', 'all'],
                        help='Which type of dataset is being processed. Default equal to r2n2. '+\
                             'To convert all obj files and if not used ShapeNet dataset, enter `all`',
                        default='r2n2')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save binvox files.')
    parser.add_argument('-n', '--num-process', type=int, 
                        help='Number of process.', default=4)
    parser.add_argument('--virtual-display', type='store_true', 
                        help='Use virtual display on headless server. Package Xvfb is used.')
    parser.add_argument('--in-unit-cube', type='store_true', 
                        help='Place object in the unit cube for voxel generation. ' +
                             'Needed if generate voxels for ShapeNet dataset to center and give all objects same size. ' +
                             'More info with examples could be found in the README.')
    parser.add_argument('--exact-generation', type='store_true', 
                        help='Exact voxelization (any voxel intersecting a convex polygon gets set). ' +
                             'When use this type of generation display not used, ' +
                             'i.e. generation could be done even without virtual display')
    args = parser.parse_args()
    main(args)
    