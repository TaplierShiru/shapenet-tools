import os
import contextlib
from math import radians
from typing import List
import numpy as np
from PIL import Image
import bpy

from constants import OPTIX, CUDA, OPENCL


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

    def __init__(
            self, generate_depth=True, generate_normal=False, generate_albedo=False, 
            gpu_count=0, gpu_ids: List[int]=None, gpu_only=False, reset_scene=False, preferred_device_type=OPTIX,
            backface_culling=False, disable_auto_polygons_smooth=True):
        if reset_scene:
            # Sometimes blender could not proper init model and drop weird errors about light
            # That its doesnt exist...
            # Dump solution here is to try again render and reset scene to default, in most cases its work as exptected
            # Reset current scene to default
            bpy.ops.wm.read_homefile(app_template="")
        use_gpu_ids = gpu_ids is not None
        if gpu_count > 0 or (use_gpu_ids and len(gpu_ids) > 0):
            # Taken from here:
            #    https://github.com/nytimes/rd-blender-docker/issues/3#issuecomment-618459326
            bpy.data.scenes['Scene'].render.engine = 'CYCLES'
            cprefs = bpy.context.preferences.addons['cycles'].preferences

            preferred_device_type = preferred_device_type.upper()
            all_devices_types = [OPTIX, CUDA, OPENCL]
            all_devices_types.remove(preferred_device_type) # Remove selected device from all types
            all_devices_types.insert(0, preferred_device_type) # Set selected device at the start aka top priority
            # Attempt to set GPU device types if available
            for compute_device_type in all_devices_types:
                try:
                    cprefs.compute_device_type = compute_device_type
                    break
                except TypeError:
                    pass
            for scene in bpy.data.scenes:
                scene.cycles.device = 'GPU'
            # get_devices() to let Blender detects GPU device
            cprefs.get_devices()
            # Find all needed GPU devices
            gpu_devices_list = []
            for device in cprefs.devices:
                # Turn on only devices which type equal to enabled one
                if device.type == cprefs.compute_device_type:
                    gpu_devices_list.append(device)
                elif device.type == 'CPU' and not gpu_only:
                    # TODO: Should CPU device also be enabled?
                    device.use = True
                else:
                    # Make sure we use only what we want
                    device.use = False

            enabled_gpu_count = 0
            for device_id, device in enumerate(sorted(gpu_devices_list, key=lambda x: x.id)):
                if (gpu_ids is not None and device_id in gpu_ids) or (not use_gpu_ids and enabled_gpu_count < gpu_count):
                    device.use = True
                    enabled_gpu_count += 1
                else:
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
        self.backface_culling = backface_culling
        self.disable_auto_polygons_smooth = disable_auto_polygons_smooth
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
            # create a file output node and set the path
            depthFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
            depthFileOutput.label = 'Depth Output'
            depthFileOutput.format.file_format = "OPEN_EXR"
            depthFileOutput.format.compression = 0 # Raw output for faster save
            depthFileOutput.format.quality = 100 # Original quality
            depthFileOutput.format.exr_codec = 'NONE' # Without codecs
            links.new(rl.outputs['Depth'], depthFileOutput.inputs[0])
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
        # TODO: Is there a way to give obj a name, or maybe somehow safer take model after load?
        # Loaded model will have name equal to filename
        _, filename = os.path.split(file_path)
        # Cut type of the file
        filename = filename.split('.')[0]

        if file_path.endswith('obj'):
            # Legacy import for ShapeNET v1, 
            # bpy.ops.import_scene.obj(filepath=file_path) 
            # There is repo that support render of v1, but not sure about v2
            #   https://github.com/DLR-RM/BlenderProc/tree/main
            # but new one used here also works except some parameters must be changed compare to v2
            bpy.ops.wm.obj_import(filepath=file_path)
        elif file_path.endswith('3ds'):
            bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
        elif file_path.endswith('dae'):
            # Must install OpenCollada.
            bpy.ops.wm.collada_import(filepath=file_path)
        else:
            raise Exception("Loading failed: %s Model loading for type %s not Implemented" %
                            (file_path, file_path[-4:]))
        
        # Below stuff is needed as ShapeNet have some wrong directions for normals
        # Solution found here:
        #   https://github.com/DLR-RM/BlenderProc/issues/634
        # But there are skipped part with smooth of normals 
        # which lead to strange black lines or wrong light on obj surface
        if self.backface_culling:
            # v1 have right materials, but v2 dont
            for mat in bpy.data.materials:
                mat.use_backface_culling = True 
        
        if self.disable_auto_polygons_smooth:
            # Better to use it for batch v1 and v2 versions
            for pol in bpy.data.objects[filename].data.polygons:
                pol.use_smooth = False
            bpy.data.objects[filename].data.use_auto_smooth = False

        if object_scale != 1.0:
            # Scale loaded model to fit into the camera properly
            bpy.data.objects[filename].scale = [object_scale] * 3

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
        base_dir = '/'.join(image_path.split('/')[:-1])
        if self.depthFileOutput:
            self.depthFileOutput.base_path = base_dir
            # Tail name with #### means that # will be replaced with number
            # I.e. _depth_0001, _depth_0002
            # In case current repo, tail will be always as _depth_0001
            # its constraint from Blender itself
            depth_filename = image_path.split('/')[-1].split('.')[-2] + '_depth_####'
            depth_filename = os.path.normpath(depth_filename)
            self.depthFileOutput.file_slots[0].path = depth_filename
            
        if self.normalFileOutput:
            self.normalFileOutput.base_path = base_dir
            normal_filename = image_path.split('/')[-1].split('.')[-2] + '_normal' + image_path.split('/')[-1].split('.')[-1]
            self.normalFileOutput.file_slots[0].path = normal_filename
            
        if self.albedoFileOutput:
            self.albedoFileOutput.base_path = base_dir
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

    def __init__(
            self, generate_depth=True, generate_normal=False, generate_albedo=False, 
            gpu_count=0, gpu_ids: List[int]=None, gpu_only=False, preferred_device_type=OPTIX, reset_scene=False,
            backface_culling=False, disable_auto_polygons_smooth=True):
        super().__init__(
            generate_depth=generate_depth, 
            generate_normal=generate_normal, 
            generate_albedo=generate_albedo,
            gpu_count=gpu_count,
            gpu_ids=gpu_ids,
            gpu_only=gpu_only,
            preferred_device_type=preferred_device_type,
            reset_scene=reset_scene,
            backface_culling=backface_culling, disable_auto_polygons_smooth=disable_auto_polygons_smooth
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
        #self.light.data.cycles.cast_shadow = False
        #self.light.data.shadow_soft_size = 0

        light_2.location       = (0, 2, 2)
        light_2.rotation_mode  = 'ZXY'
        light_2.rotation_euler = (-radians(45), 0, radians(90))
        light_2.data.energy = 0.7
        #light_2.data.cycles.cast_shadow = False
        #light_2.data.shadow_soft_size = 0

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
