import argparse
import traceback
import os
import contextlib
from math import radians
import numpy as np
from PIL import Image
from tempfile import TemporaryFile
from contextlib import contextmanager
import bpy
import sys
from multiprocessing import Pool


R2N2_MODELS = [
    '02691156', '02933112', '03001627', '03636649', 
    '04090263', '04379243', '04530566', '02828884', 
    '02958343', '03211117', '03691459', '04256520', 
    '04401088'  
]

ALL_SHAPENET_MODELS = [
    '02691156', '02808440', '02871439', '02933112', 
    '02958343', '03085013', '03325088', '03593526', 
    '03691459', '03790512', '03948459', '04090263', 
    '04330267', '04468005', '02747177', '02818832', 
    '02876657', '02942699', '02992529', '03207941', 
    '03337140', '03624134', '03710193', '03797390', 
    '03991062', '04099429', '04379243', '04530566',
    '02773838', '02828884', '02880940', '02946921', 
    '03001627', '03211117', '03467517', '03636649', 
    '03759954', '03928116', '04004475', '04225987', 
    '04401088', '04554684', '02801938', '02843684', 
    '02924116', '02954340', '03046257', '03261776', 
    '03513137', '03642806', '03761084', '03938244', 
    '04074963', '04256520', '04460130',
]


@contextmanager
def stdout_redirected(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout


def voxel2mesh(voxels):
    cube_verts = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2],
                  [1, 3, 2],
                  [2, 3, 6],
                  [3, 7, 6],
                  [0, 2, 6],
                  [0, 6, 4],
                  [0, 5, 1],
                  [0, 4, 5],
                  [6, 7, 5],
                  [6, 5, 4],
                  [1, 7, 3],
                  [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    l, m, n = voxels.shape

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0
    for i in range(l):
        for j in range(m):
            for k in range(n):
                # If there is a non-empty voxel
                if voxels[i, j, k] > 0:
                    verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
                    faces.extend(cube_faces + curr_vert)
                    curr_vert += len(cube_verts)

    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


class BaseRenderer:
    model_idx   = 0

    def __init__(self, generate_depth=True, generate_normal=False, generate_albedo=False, gpu_count=0):
        if gpu_count > 0:
            # Taken from here:
            #    https://github.com/nytimes/rd-blender-docker/issues/3#issuecomment-618459326
            bpy.data.scenes['Scene'].render.engine = 'CYCLES'
            cprefs = bpy.context.preferences.addons['cycles'].preferences
            # Attempt to set GPU device types if available
            for compute_device_type in ('OPTIX', 'CUDA', 'OPENCL'):
                try:
                    cprefs.compute_device_type = compute_device_type
                    break
                except TypeError:
                    pass
            for scene in bpy.data.scenes:
                scene.cycles.device = 'GPU'
            # get_devices() to let Blender detects GPU device
            cprefs.get_devices()
            # Enable all GPU devices
            enabled_gpu_count = 0
            for device in cprefs.devices:
                # TODO: Should CPU device also be enabled?
                # Turn on only devices which type equal to enabled one
                if device.type == cprefs.compute_device_type and enabled_gpu_count < gpu_count:
                    device.use = True
                    enabled_gpu_count += 1
                else:
                    # Make sure we use only what we want
                    device.use = False
        
        # Setup additional view layers
        self.setupAdditionalViewLayers(generate_depth, generate_normal, generate_albedo)
        # remove the default cube
        bpy.ops.object.select_pattern(pattern="Cube")
        bpy.ops.object.delete()

        render_context = bpy.context.scene.render
        world  = bpy.context.scene.world
        camera = bpy.data.objects['Camera']
        light_1  = bpy.data.objects['Light']
        light_1.data.type = 'SUN'

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera.location       = (1, 0, 0)
        camera.rotation_mode  = 'ZXY'
        camera.rotation_euler = (0, radians(90), radians(90))

        # parent camera with a empty object at origin
        org_obj                = bpy.data.objects.new("RotCenter", None)
        org_obj.location       = (0, 0, 0)
        org_obj.rotation_euler = (0, 0, 0)
        bpy.context.collection.objects.link(org_obj)

        camera.parent = org_obj  # setup parenting

        # render setting
        render_context.resolution_percentage = 100
        world.color = (1, 1, 1)  # set background color to be white

        self.render_context = render_context
        self.org_obj = org_obj
        self.camera = camera
        self.light = light_1
        self._set_lighting()
    
    def setupAdditionalViewLayers(self, generate_depth=True, generate_normal=False, generate_albedo=False):
        self.depthFileOutput = None
        self.depth_map = None
        
        self.normalFileOutput = None
        self.albedoFileOutput = None
        
        # Set up rendering of depth map:
        bpy.context.scene.render.use_compositing = True
        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_layers[0].use_pass_z = True
        tree = bpy.context.scene.node_tree
        links = tree.links
        
        # Add passes for additionally dumping albed and normals.
        if generate_normal:
            bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = True
        if generate_albedo:
            bpy.context.scene.view_layers['ViewLayer'].use_pass_diffuse_color = True

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')
        
        if generate_depth:
            map = tree.nodes.new(type="CompositorNodeMapValue")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with
            # resulting depth map.
            map.offset = [-1]
            # map.size = [depth_scale]
            map.use_min = True
            map.min = [0]
            map.use_max = True
            map.max = [255]
            self.depth_map = map
            try:
                links.new(rl.outputs['Depth'], self.depth_map.inputs[0])
            except KeyError:
                # TODO: Some versions of blender don't like this?
                print('Error! Cant link with Z dim for depth')

            invert = tree.nodes.new(type="CompositorNodeInvert")
            links.new(map.outputs[0], invert.inputs[1])

            # create a file output node and set the path
            depthFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
            depthFileOutput.label = 'Depth Output'
            links.new(invert.outputs[0], depthFileOutput.inputs[0])
            self.depthFileOutput = depthFileOutput
        
        if generate_normal:
            scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
            scale_normal.blend_type = 'MULTIPLY'
            scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
            links.new(rl.outputs['Normal'], scale_normal.inputs[1])

            bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
            bias_normal.blend_type = 'ADD'
            bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
            links.new(scale_normal.outputs[0], bias_normal.inputs[1])

            normalFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
            normalFileOutput.label = 'Normal Output'
            links.new(bias_normal.outputs[0], normalFileOutput.inputs[0])
            self.normalFileOutput = normalFileOutput
        
        if generate_albedo:
            albedoFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
            albedoFileOutput.label = 'Albedo Output'
            links.new(rl.outputs['DiffCol'], albedoFileOutput.inputs[0])
            self.albedoFileOutput = albedoFileOutput
    
    def initialize(self, models_fn, viewport_size_x, viewport_size_y):
        self.models_fn = models_fn
        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

    def _set_lighting(self):
        pass

    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov, max_camera_dist=0):
        # TODO: Use fov. In the original r2n2 render code, fov also doesnt used.
        #       With current setup its would be same as in the r2n2 render code.
        self.org_obj.rotation_euler = (0, 0, 0)
        self.light.location = (
            distance_ratio *(max_camera_dist + 2), 
            0, 0
        )
        self.camera.location = (
            distance_ratio * max_camera_dist, 
            0, 0
        )
        self.org_obj.rotation_euler = (
            radians(-yaw),
            radians(-altitude),
            radians(-azimuth)
        )

        if self.depth_map:
            self.depth_map.offset = [
                (-1) * distance_ratio * max_camera_dist
            ]

    def setTransparency(self, transparency):
        """ 
        transparency is either 'SKY', 'TRANSPARENT'
        If set 'SKY', render background using sky color.
        
        """
        # 0 for "SKY"; 1 for "TRANSPARENT
        self.render_context.film_transparent = transparency == 'TRANSPARENT'

    def selectModel(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern="RotCenter")
        bpy.ops.object.select_pattern(pattern="Light*")
        bpy.ops.object.select_pattern(pattern="Camera")
        bpy.ops.object.select_all(action='INVERT')

    def printSelection(self):
        print(bpy.context.selected_objects)

    def clearModel(self):
        self.selectModel()
        bpy.ops.object.delete()

        # The meshes still present after delete
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)

    def setModelIndex(self, model_idx):
        self.model_idx = model_idx

    def loadModel(self, file_path=None, object_scale=1.0):
        if file_path is None:
            file_path = self.models_fn[self.model_idx]
        
        if file_path.endswith('obj'):
            bpy.ops.import_scene.obj(filepath=file_path)
            for i, mat in enumerate(bpy.data.materials):
                mat.use_backface_culling = False 
        elif file_path.endswith('3ds'):
            bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
        elif file_path.endswith('dae'):
            # Must install OpenCollada.
            bpy.ops.wm.collada_import(filepath=file_path)
        else:
            raise Exception("Loading failed: %s Model loading for type %s not Implemented" %
                            (file_path, file_path[-4:]))
        if object_scale != 1.0:
            # Scale loaded model to fit into the camera properly
            # TODO: Is there a way to give obj a name, or maybe somehow safer take model after load?
            # Loaded model will have name equal to filename
            _, filename = os.path.split(file_path)
            # Cut type of the file
            filename = filename.split('.')[0]
            for k in bpy.data.objects.keys():
                if filename in k:
                    bpy.data.objects[k].scale = [object_scale] * 3

    def render(self, load_model=True, clear_model=True, resize_ratio=None,
               return_image=True, image_path=os.path.join('./tmp', 'tmp.png')):
        """ Render the object """
        if load_model:
            self.loadModel()
        # resize object
        self.selectModel()
        if resize_ratio:
            bpy.ops.transform.resize(value=resize_ratio)

        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        if self.depthFileOutput:
            base_dir = '/'.join(image_path.split('/')[:-1])
            self.depthFileOutput.base_path = base_dir
            # Tail name with #### means that # will be replaced with number
            # I.e. _depth_0001, _depth_0002
            # In case current repo, tail will be always as _depth_0001
            # its constraint from Blender itself
            depth_filename = image_path.split('/')[-1].split('.')[-2] + '_depth_####'
            depth_filename = os.path.normpath(depth_filename)
            self.depthFileOutput.file_slots[0].path = depth_filename
            
        if self.normalFileOutput:
            # Clear base path, otherwise it wouldnt be saved where we want it
            self.normalFileOutput.base_path = ''
            
            normal_filename = image_path.split('/')[-1].split('.')[-2] + '_normal' + image_path.split('/')[-1].split('.')[-1]
            self.normalFileOutput.file_slots[0].path = normal_filename
            
        if self.albedoFileOutput:
            # Clear base path, otherwise it wouldnt be saved where we want it
            self.albedoFileOutput.base_path = ''
            albedo_filename = image_path.split('/')[-1].split('.')[-2] + '_albedo' + image_path.split('/')[-1].split('.')[-1]
            self.albedoFileOutput.file_slots[0].path = albedo_filename
            
        bpy.ops.render.render(write_still=True)  # save straight to file
        if resize_ratio:
            bpy.ops.transform.resize(value=(
                1/resize_ratio[0],
                1/resize_ratio[1], 
                1/resize_ratio[2],
            ))
        if clear_model:
            self.clearModel()

        if return_image:
            im = np.array(Image.open(self.result_fn))  # read the image

            # Last channel is the alpha channel (transparency)
            return im[:, :, :3], im[:, :, 3]


class ShapeNetRenderer(BaseRenderer):

    def __init__(self, generate_depth=True, generate_normal=False, generate_albedo=False, gpu_count=0):
        super().__init__(
            generate_depth=generate_depth, 
            generate_normal=generate_normal, 
            generate_albedo=generate_albedo,
            gpu_count=gpu_count,
        )
        self.setTransparency('TRANSPARENT')

    def _set_lighting(self):
        # Create new light datablock
        light_data = bpy.data.lights.new(name="New Light", type='SUN')

        # Create new object with our light datablock
        light_2 = bpy.data.objects.new(name="New Light", object_data=light_data)
        bpy.context.collection.objects.link(light_2)

        # put the light behind the camera. Reduce specular lighting
        self.light.location       = (0, -2, 2)
        self.light.rotation_mode  = 'ZXY'
        self.light.rotation_euler = (radians(45), 0, radians(90))
        self.light.data.energy = 0.7

        light_2.location       = (0, 2, 2)
        light_2.rotation_mode  = 'ZXY'
        light_2.rotation_euler = (-radians(45), 0, radians(90))
        light_2.data.energy = 0.7

# TODO: Test it
# NOTICE! Not tested here because for now not needed here
class VoxelRenderer(BaseRenderer):

    def __init__(self, generate_depth=True, generate_normal=False, generate_albedo=False):
        super().__init__(
            generate_depth=generate_depth, 
            generate_normal=generate_normal, 
            generate_albedo=generate_albedo
        )
        self.setTransparency('SKY')

    def _set_lighting(self):
        self.light.location       = (0, 3, 3)
        self.light.rotation_mode  = 'ZXY'
        self.light.rotation_euler = (-radians(45), 0, radians(90))
        self.light.data.energy = 0.7

        # Create new light datablock
        light_data = bpy.data.lights.new(name="New Light", type='SUN')

        # Create new object with our lamp datablock
        light_2 = bpy.data.objects.new(name="New Light", object_data=light_data)
        bpy.context.collection.objects.link(light_2)

        light_2.location       = (4, 1, 6)
        light_2.rotation_mode  = 'XYZ'
        light_2.rotation_euler = (radians(37), radians(3), radians(106))
        light_2.data.energy = 0.7

    def render_voxel(self, pred, thresh=0.4,
                     image_path=os.path.join('./tmp', 'tmp.png')):
        # Cleanup the scene
        self.clearModel()
        out_f = os.path.join('./tmp', 'tmp.obj')
        occupancy = pred > thresh
        vertices, faces = voxel2mesh(occupancy)
        with contextlib.suppress(IOError):
            os.remove(out_f)
        write_obj(out_f, vertices, faces)

        # Load the obj
        bpy.ops.import_scene.obj(filepath=out_f)
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # save straight to file

        im = np.array(Image.open(image_path))  # read the image

        # Last channel is the alpha channel (transparency)
        return im[:, :, :3], im[:, :, 3]

    
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
        generate_depth: bool, gpu_count=0, overwrite=False, load_attempt=0):
    last_render_file_path = os.path.join(
        save_dir, category,
        curr_model_id, f'{str(num_rendering-1).zfill(2)}.png'
    )
    if os.path.isfile(last_render_file_path) and not overwrite:
        return

    try:
        try:
            renderer = ShapeNetRenderer(generate_depth=generate_depth, gpu_count=gpu_count)
        except KeyError as e:
            # Sometimes blender could not proper init model and drop weird errors about light
            # That its doesnt exist...
            # Dump solution here is to try again render and reset scene to default, in most cases its work as exptected
            # Reset current scene to default
            bpy.ops.wm.read_homefile(app_template="")
            # Try again
            renderer = ShapeNetRenderer(generate_depth=generate_depth, gpu_count=gpu_count)
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

            image_path = os.path.join(save_dir, category, curr_model_id, f'{str(j).zfill(2)}.png')

            # This with state will silent all prints from blender into temporary file
            # TODO: Its seems that this hack doesnt work and prints anyway printed
            with TemporaryFile() as f, stdout_redirected(f):
                renderer.render(
                    load_model=False, return_image=False,
                    clear_model=False, image_path=image_path
                )
        renderer.clearModel()
        return True
    except Exception as e:
        traceback.print_exc()
        print('='*20)
        print(
            f'Something go wrong while render: category={category}, curr_model_id={curr_model_id}. ',
        )
        print('='*20)
        return False


def test_render(
        save_dir: str, file_path: str, category: str, curr_model_id: str, 
        width: int, height: int, num_rendering: int,
        max_camera_dist: float, object_scale: float,
        generate_depth: bool, gpu_count=0):
    renderer = ShapeNetRenderer(generate_depth=generate_depth, gpu_count=gpu_count)
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

        image_path = os.path.join(save_dir, category, curr_model_id, f'{str(j).zfill(2)}.png')

        renderer.render(
            load_model=False, return_image=False,
            clear_model=False, image_path=image_path
        )
    renderer.clearModel()


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
            args.depth, args.gpu_count
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
            args.width, args.height, args.num_rendering, 
            args.max_camera_dist, args.object_scale, 
            args.depth, args.gpu_count
        )
            for category in models
            for model_id in os.listdir(os.path.join(args.base_dir, category))
    ]

    render_args_batches = [
        render_args[i * args.batch_size: min((i + 1) * args.batch_size, len(render_args))]
        for i in range(len(render_args) // args.batch_size + 1)
    ]

    for render_args_batch in render_args_batches:
        with Pool(processes=args.num_process) as pool:
            pool.starmap(render_model, render_args_batch)
    
    
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
                        'Zero GPU count means render only via CPU.', default=1)
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite images if they were created. ')
    parser.add_argument('--generate-type', choices=['r2n2', 'all', 'custom'], 
                        help='Type of dataset generation. ' 
                        'R2n2 generate only 13 objects, while all generate all objects from ShapeNet. '
                        'If custom is provided, when parameter -m must be also provided with args which object to generate',
                        default='r2n2')
    parser.add_argument('-m', '--models', nargs='*',
                        help='If generation type is custom when list of models to generate will be taken from here')
    args = parser.parse_args()
    main(args)
    