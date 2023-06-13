import argparse
import traceback
import os
from typing import List
import numpy as np
from contextlib import contextmanager
import sys
from multiprocessing import Process

import time

from constants import FOLDER_SAVE_NAME, R2N2_MODELS, ALL_SHAPENET_MODELS, OPTIX, CUDA, OPENCL


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    How to use it:
    ```
        import os

        with stdout_redirected(to=filename):
            print("from Python")
            os.system("echo non-Python applications are also supported")
        # Or to skip any print
        with stdout_redirected():
            print("from Python")
            os.system("echo non-Python applications are also supported")
    ```

    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

    
def append_path_to_model(path: str, version_dataset: int):
    if version_dataset == 1:
        return f'{path}/model.obj'
    elif version_dataset == 2:
        return f'{path}/models/model_normalized.obj'
    else:
        raise Exception(f'Unknown version={version_dataset}')

    
def render_model(
        save_dir: str, file_path: str, category: str, curr_model_id: str, 
        width: int, height: int, num_rendering: int, 
        max_camera_dist: float, object_scale: float,
        generate_depth: bool, gpu_count=0, gpu_ids: List[int]=None,
        gpu_only=False, preferred_device_type=OPTIX, is_debug=False):
    from render_blender import ShapeNetRenderer

    try:
        start_time = time.time()
        renderer = ShapeNetRenderer(
            generate_depth=generate_depth, gpu_count=gpu_count, gpu_ids=gpu_ids,
            gpu_only=gpu_only, preferred_device_type=preferred_device_type
        )
        renderer.initialize([file_path], width, height)
        renderer.loadModel(object_scale=object_scale)

        for j in range(num_rendering):
            # Set random viewpoint.
            az, el, depth_ratio = list(
                *([360, 5, 0.3] * np.random.rand(1, 3) + [0, 25, 0.65]))
            renderer.setViewpoint(
                az, el, 0, depth_ratio, 25, 
                max_camera_dist=max_camera_dist
            )

            image_path = os.path.join(save_dir, category, curr_model_id, FOLDER_SAVE_NAME, f'{str(j).zfill(2)}.png')
            # This state will silent all prints from blender
            # Its possible to direct these prints to some file, but they do not needed here
            with stdout_redirected():
                renderer.render(
                    load_model=False, return_image=False,
                    clear_model=False, image_path=image_path
                )
        renderer.clearModel()
        end_time = time.time()
        if is_debug:
            print(f'Time passed: {end_time - start_time}')
        return True
    except Exception as e:
        traceback.print_exc()
        print('='*20)
        print(
            f'Something go wrong while render: category={category}, curr_model_id={curr_model_id}. ',
        )
        print('='*20)
        return False


def start_render_model_single_process(render_args_batch: List[tuple], 
        width: int, height: int, num_rendering: int, 
        max_camera_dist: float, object_scale: float,
        generate_depth: bool, gpu_count=0, gpu_ids: List[int]=None,
        gpu_only=False, preferred_device_type=OPTIX, is_debug=False):
    
    for arg_batch in render_args_batch:
        render_model(*arg_batch, 
            width=width, height=height, num_rendering=num_rendering, 
            max_camera_dist=max_camera_dist, object_scale=object_scale,
            generate_depth=generate_depth, gpu_count=gpu_count, gpu_ids=gpu_ids,
            gpu_only=gpu_only, preferred_device_type=preferred_device_type, is_debug=is_debug
        )


def test_render(
        save_dir: str, file_path: str, category: str, curr_model_id: str, 
        width: int, height: int, num_rendering: int,
        max_camera_dist: float, object_scale: float,
        generate_depth: bool, gpu_count=0, gpu_ids: List[int]=None,
        gpu_only=False, preferred_device_type=OPTIX):
    from render_blender import ShapeNetRenderer
    start_time = time.time()
    renderer = ShapeNetRenderer(
        generate_depth=generate_depth, gpu_count=gpu_count, gpu_ids=gpu_ids,
        gpu_only=gpu_only, preferred_device_type=preferred_device_type,
    )
    renderer.initialize([file_path], width, height)
    renderer.loadModel(object_scale=object_scale)
    
    for j in range(num_rendering):
        # Set random viewpoint.
        az, el, depth_ratio = list(
            *([360, 5, 0.3] * np.random.rand(1, 3) + [0, 25, 0.65]))
        renderer.setViewpoint(
            az, el, 0, depth_ratio, 25, 
            max_camera_dist=max_camera_dist
        )

        image_path = os.path.join(save_dir, category, curr_model_id, FOLDER_SAVE_NAME, f'{str(j).zfill(2)}.png')

        renderer.render(
            load_model=False, return_image=False,
            clear_model=False, image_path=image_path
        )
    renderer.clearModel()
    end_time = time.time()
    print(f'Time passed: {end_time - start_time}')


def main(args):    
    if args.save_folder:
        save_dir = args.save_folder
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = args.base_dir
        
    if args.test:
        category = '02958343'
        model_id = '63599f1dc1511c25d76439fb95cdd2ed'
        test_render(
            save_dir, 
            append_path_to_model(os.path.join(args.base_dir, category, model_id), args.dataset_version), 
            category, model_id, 
            args.width, args.height, args.num_rendering, 
            args.max_camera_dist, args.object_scale, 
            args.depth, args.gpu_count, args.gpu_ids,
            args.gpu_only, args.preferred_device_type
        )
        return
    if args.generate_type == 'r2n2':
        models = R2N2_MODELS
    elif args.generate_type == 'all':
        models = ALL_SHAPENET_MODELS
    elif args.generate_type == 'custom':
        models = args.models
        if len(models) == 0:
            raise Exception('Generation type is custom, but size of the list of models equal to zero.')
    else:
        raise Exception(f'Unknown generate type = {args.generate_type}')
    
    # TODO: Skip bad models (aka empty folders\missing texture and etc). 
    #       But the authors in the repo from link below noted that these examples could be less than 200 in sum,
    #       So its not a big deal to hurry fix it\implement.
    # There are several models which are corrupted or missing some texture, full list can be found here
    #     https://github.com/google-research/kubric/blob/main/shapenet2kubric/shapenet_denylist.py

    # Multi-process pre-rendering
    # Blender tends to get slower after it renders few hundred models. Start
    # over the whole pool every BATCH_SIZE models to boost the speed.
    render_args = [
        (
            save_dir, 
            append_path_to_model(os.path.join(args.base_dir, category, model_id), args.dataset_version), 
            category, model_id, 
        )
            for category in models
            for model_id in os.listdir(os.path.join(args.base_dir, category))
    ]
    # If last img exist - then all others also, so just skip this obj file
    print(f'Number of models to render equal to {len(render_args)}')
    render_args = list(filter(
        lambda in_args: not os.path.isfile(os.path.join(
            in_args[0], in_args[2],
            in_args[3], FOLDER_SAVE_NAME, 
            f'{str(args.num_rendering-1).zfill(2)}.png'
            )) or args.overwrite, 
        render_args
    ))
    print(f'After checking for existing ones number of models to render equal to {len(render_args)}')
    render_args_batches = [
        render_args[i * args.batch_size: min((i + 1) * args.batch_size, len(render_args))]
        for i in range(len(render_args) // args.batch_size + 1)
    ]

    # Get id for GPU
    def assign_gpu_id(i: int):
        if args.gpu_uniform_id:
            if args.gpu_ids is None:
                # If ids not provided just give id assuming that count will be max id value
                return [i % args.gpu_count]
            # Otheriwse take id
            return [args.gpu_ids[i % len(args.gpu_ids)]]
        
        return args.gpu_ids

    for render_args_batch in render_args_batches:
        workers = [
            Process(
                target=start_render_model_single_process, 
                args = (
                    render_args_batch, 
                    args.width, args.height, args.num_rendering, 
                    args.max_camera_dist, args.object_scale, 
                    args.depth, args.gpu_count, assign_gpu_id(i),
                    args.gpu_only, args.preferred_device_type,
                    args.debug
                )
            ) 
            for i in range(args.num_process)
        ]
        for p in workers:
            p.start()

        for p in workers:
            p.join()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=str,
                        help='Path to ShapeNet dataset folder.')
    parser.add_argument('-s', '--save-folder', type=str,
                        help='Path to save rendered images.')
    parser.add_argument('-d', '--dataset-version', choices=[1, 2], 
                        help='Choose version of ShapeNet dataset.', default=2)
    parser.add_argument('-t', '--test', action='store_true',
                        help='Generate test view from folder with id `02958343` and objects id `63599f1dc1511c25d76439fb95cd`.')
    parser.add_argument('-b', '--batch-size', type=int, 
                        help='How many objects to process in a single process', default=128)
    parser.add_argument('-n', '--num-process', type=int, 
                        help='Number of process.', default=4)
    parser.add_argument('--width', type=int, 
                        help='Width of rendered images.', default=127)
    parser.add_argument('--height', type=int, 
                        help='Height of rendered images.', default=127)
    parser.add_argument('--num_rendering', type=int, 
                        help='Number of renderings view per object.', default=24)
    parser.add_argument('--max-camera-dist', type=float, 
                        help='Maximum camera distance. '
                        'For ShapeNet (and almost every other object) its better to live it with default value.', 
                        default=1.75)
    parser.add_argument('--object-scale', type=float, 
                        help='Scale of the object. '
                        'If value equal to 1.0, when on the ShapeNet dataset models are very close to the camera '
                        'and final render do not corresponds to what is in r2n2 rendered dataset. '
                        'Value by default will give almost equal results as in r2n2 renders.', 
                        default=0.57)
    parser.add_argument('--depth', action='store_true',
                        help='Generate depth per view. Image view and corresponding depth will had same name, '
                        'except name for the depth will be with tail `_depth_0001.png` '
                        '(this name will be always same for every view). ' 
                        'Its some sort of constraints from the blender itself. ')
    parser.add_argument('--gpu-count', type=int, 
                        help='How many GPUs is to use for render. GPU computation could be done in several types, '
                        'in this situation order of attempts to set device type: OptiX, CUDA, OpenCL. '
                        'Zero GPU count means render only via CPU. The ids of the used GPUs start from zero to your gpu counter.', default=1)
    parser.add_argument('--gpu-only', action='store_true',
                        help='By default GPU will be used with CPU. With this parameter only GPU will be used.')
    parser.add_argument('--preferred-device-type', choices=[OPTIX, CUDA, OPENCL], default=OPTIX,
                        help=f'Type of device for GPUs. Available: {OPTIX} (default), {CUDA}, {OPENCL}.')
    parser.add_argument('--gpu-ids', nargs='*', default=None,
                        help='Ids of the gpu from zero to number of GPUs, which will be used by Blender. Example: 0 2 3.'
                        'NOTICE! This parameter has a higher priority than gpu-counter which will be ignored.')
    parser.add_argument('--gpu-uniform-id', action='store_true', 
                        help='If GPU ids are provided or gpu counter more than 1, when with this parameter ids will be uniform distributed for all processors, '
                        'otherwise (by default) every process will use provided gpu ids without change.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite images if they were created. ')
    parser.add_argument('--generate-type', choices=['r2n2', 'all', 'custom'], 
                        help='Type of dataset generation. ' 
                        'R2n2 generate only 13 objects, while all generate all objects from ShapeNet. '
                        'If custom is provided, when parameter -m must be also provided with args which object to generate',
                        default='r2n2')
    parser.add_argument('-m', '--models', nargs='*',
                        help='If generation type is custom when list of models to generate will be taken from here')
    parser.add_argument('--debug', action='store_true',
                        help='Additional prints will be shown to debug and compare speed.')
    args = parser.parse_args()

    if args.gpu_ids is not None:
        args.gpu_ids = list(map(int, args.gpu_ids))

    main(args)
    