import os
import argparse
import glob


CONST_R2N2_DATASET = [
    '02691156', '02933112', '03001627', '03636649', 
    '04090263', '04379243', '04530566', '02828884', 
    '02958343', '03211117', '03691459', '04256520', 
    '04401088'  
]


def main(args):
    for category in os.listdir(args.binvox_generated_path):
        binvox_generated_category_path = os.path.join(args.binvox_generated_path, category)
        rendered_imgs_category_path = os.path.join(args.rendered_imgs_path, category)

        if not os.path.isdir(rendered_imgs_category_path):
            print(f'Folder {rendered_imgs_category_path} does not exist.')
            continue
            
        # Compare content of two folders
        for model_id in os.listdir(binvox_generated_category_path):
            binvox_generated_model_path = os.path.join(binvox_generated_category_path, model_id)
            rendered_imgs_model_path = os.path.join(rendered_imgs_category_path, model_id)

            if not os.path.isdir(rendered_imgs_model_path):
                print(f'Folder {rendered_imgs_model_path} does not exist.')
                continue
            
            binvox_path = os.path.join(binvox_generated_model_path, 'models', 'model_normalized.binvox')
            if not os.path.isfile(binvox_path):
                print(f'Binvox file in {binvox_path} does not exist.')
            
            rendered_imgs_path = os.path.join(binvox_generated_model_path, 'rendering')
            num_imgs = len(glob.glob(os.path.join(rendered_imgs_path, '??.png'))) # png could be also be images for depth
            if num_imgs != args.num_rendering:
                print(f'In folder {rendered_imgs_path} not all rendering images complete. '
                      f'In the folder {num_imgs} images but {args.num_rendering} expected. '
                )
            if args.depth:
                num_depth_imgs = len(glob.glob(os.path.join(rendered_imgs_path, '??_depth_0001.png')))
                if num_depth_imgs != args.num_rendering:
                    print(f'In folder {rendered_imgs_path} not all rendering depth images complete. '
                        f'In the folder {num_depth_imgs} depth images but {args.num_rendering} expected. '
                    )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('binvox_generated_path', type=str,
                        help='Path to folder with binvox generated.')
    parser.add_argument('rendered_imgs_path', type=str,
                        help='Path to folder with rendered images by Blender.')
    parser.add_argument('--depth', action='store_true',
                        help='Use generated depth per view and store to the h5 file. ')
    parser.add_argument('--num_rendering', type=int, 
                    help='Number of renderings view per object.', default=24)
    args = parser.parse_args()
    main(args)
    