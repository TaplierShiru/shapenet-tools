# ShapeNet Tools
The main purpose of this repo is to provide different tools to work with ShapeNet dataset.

Implemented tools (modules):
1. [Voxelization via Binvox](#voxelization)
2. [Render of the model view via Blender](#blender-model-render)
3. [Point sample](#point-sample)
4. [H5Files creation](#h5files-creation)

Each module work independent of others tools except `utils` where placed shared code.

## Test data
The bunny object taken from [here](https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj).

All examples how these tools can be used will be shown on this bunny. All results from each module will be in corresponding folder.

## Requirements
In each module\folder you can find `requirements.txt` file with specific packages if unless explicitly stated what is needed in advance.


## Voxelization
For voxelization [Binvox](https://www.patrickmin.com/binvox/) is used. In order to start with scripts inside this module, download corresponding program from origin site of this software: https://www.patrickmin.com/binvox/. Place it inside `voxelization` folder.

You may need to install:
```bash
apt update && apt install libxmu6

apt-get update && apt-get install xvfb # Virtual display
```

And also do not forget to get execute rights for the `binvox` (for example `chmod +x binvox`).

How to use code in the folder `voxelization` on ShapeNet dataset:
```bash
python3 generate_voxels_binvox.py /path/to/ShapeNetV2 \
  -s /path/to/save/voxelized/models \ 
  -n 6 --in-unit-cube --exact-generation
```

Be default will be generated voxels from classes same as in work [3D-R2N2](https://github.com/chrischoy/3D-R2N2/tree/master). To generate from all categories, add `-d all`.

If generation is on headless server, then virtual display via `--virtual-display`could be used.

There is also parameter `--sh1-to-sh2-coords` - by default objects from ShapeNetV2 have different orientetion compare to ShapeNetV1. This parameter will transform ShapeNetV1 binvoxes to same coordinates as in ShapeNetV2.

Example of how voxelize bunny can be found in [this notebook](./voxelization/test.ipynb). Notice that bunny and most others objects (not from ShapeNet) are better to generated without `--in-unit-cube` parameter, which in most cases will only be suitable for ShapeNet dataset. Also usage of the parameter `--exact-generation` is recommended because final voxel will be more accurate with this parameter.

### Why unit-cube is needed?
Here are examples of voxel generation with different parameters, firstly without `-e` in binvox call:

<table>
  <tr>
    <th><center>./binvox -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -d 32 ./shapenet_v2.obj</center></th>
    <th><center>./binvox -d 32 ./shapenet_v2.obj</center></th>
  </tr>
  <tr>
    <td><img src="./images/sofa_box.png"></td>
    <td><img src="./images/sofa_nobox.png"></td>
  </tr>
</table>

With `-e`:

<table>
  <tr>
    <th><center>./binvox -e -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -d 32 ./shapenet_v2.obj</center></th>
    <th><center>./binvox -e -d 32 ./shapenet_v2.obj</center></th>
  </tr>
  <tr>
    <td><img src="./images/sofa_ebox.png"></td>
    <td><img src="./images/sofa_e.png"></td>
  </tr>
</table>

As you can see, with `-bb` parameters model is centered and with smaller size compare to voxel without this parameter. So, without it different models could be in different positions and with different scales, which could be bad for training of the neural network.

NOTICE! That unit cube only could be working for ShapeNet data, and for example on `bunny.obj` results are could be different, see examples below.

Here are view results for bunny below command will be shown filename in `data` folder. Without `-e`:
<table>
  <tr>
    <th>
      <center>"./binvox -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -d 32 ./bunny.obj"</center>
      <center>data/bunny_box.binvox</center>
    </th>
    <th>
      <center>"./binvox -d 32 ./bunny.obj"</center>
      <center>data/bunny_nobox.binvox</center>
    </th>
  </tr>
  <tr>
    <td><img src="./images/bunny_box.png"></td>
    <td><img src="./images/bunny_nobox.png"></td>
  </tr>
</table>

With `-e`:
<table>
  <tr>
    <th>
      <center>"./binvox -e -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -d 32 ./bunny.obj"</center>
      <center>data/bunny_ebox.binvox</center>
    </th>
    <th>
      <center>"./binvox -e -d 32 ./bunny.obj"</center>
      <center>data/bunny_e.binvox</center>
    </th>
  </tr>
  <tr>
    <td><img src="./images/bunny_ebox.png"></td>
    <td><img src="./images/bunny_e.png"></td>
  </tr>
</table>

If we look at the scale and translation for voxel without `-bb`, parameters are:
- Scale: 0.155159
- Translation: [-0.0943804, 0.0333099, -0.0616792]

As we can see from these parameters, that we should "zoom" bunny mesh in order to it fill full space of the voxel cube. If we force it to be in unit cube, then bunny will be very small and will affect on the final voxel quality (and for model training).


## Blender model render
Current module render each model with different random 24 views (number could be changed).

Blender version: ***3.5.0.*** 

In order to use Blender with GPU (CUDA\OPTIX) and use it in Python we must build from source Blender with corresponding parameters. Dockerfile could be find [here](https://gist.github.com/TaplierShiru/2b85e422703976aa1f9ec45db2ec5069) with comments. In the comments you could find how build it only for CPU, but it could be slower and very long if you want to generate views for ShapeNet dataset. 

How to use code in the folder `render_blender` on ShapeNet dataset:
```bash
python3 render_blender_main.py /path/to/ShapeNetV2 \
  -s /path/to/save/views -b 200 -n 6 --gpu-count 1
```

If you don't have gpu, just insert `--gpu-count 0` then only CPU will be used. Also this parameter support more than one GPU, i.e. next commands are possible: `--gpu-count 2`, `--gpu-count 3` and etc... By default if more than one gpu is provided when all provded GPUs will be used in every process, which are not sufficient and not fast. To enable ids of the GPus to be uniformed distributed between all process add parameter `--gpu-uniform-id`. So, as an example:
```bash
python3 render_blender_main.py /path/to/ShapeNetV2 \
  -s /path/to/save/views -b 200 -n 8 --gpu-count 4 --gpu-uniform-id
```

I found that single gpu for single process much faster than usage of several GPUs for single process. As a proof I test it on category car (02958343) and model id 63599f1dc1511c25d76439fb95cdd2ed, and here are results in the table:
<table>
  <tr>
    <th>
      <center>Configuration</center>
    </th>
    <th>
      <center>Device type</center>
    </th>
    <th>
      <center>GPU counter</center>
    </th>
    <th>
      <center>Time (seconds)</center>
    </th>
    <th>
      <center>Command</center>
    </th>
  </tr>
  <tr>
    <td>GPU + CPU</td>
    <td>OPTIX</td>
    <td>4</td>
    <td>55</td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 4 --preferred-device-type OPTIX -t --debug</code></td>
  </tr>
  <tr style="background: grey">
    <td>GPU + CPU</td>
    <td>OPTIX</td>
    <td>1</td>
    <td><b>30</b></td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 1 --preferred-device-type OPTIX -t --debug</code></td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>OPTIX</td>
    <td>4</td>
    <td>57</td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 4 --preferred-device-type OPTIX --gpu-only -t --debug</code></td>
  </tr>
  <tr style="background: grey">
    <td>GPU</td>
    <td>OPTIX</td>
    <td>1</td>
    <td><b>29</b></td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 1 --preferred-device-type OPTIX --gpu-only -t --debug</code></td>
  </tr>
  
  <tr>
    <td>GPU + CPU</td>
    <td>CUDA</td>
    <td>4</td>
    <td>50</td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 4 --preferred-device-type CUDA -t --debug</code></td>
  </tr>
  <tr style="background: grey">
    <td>GPU + CPU</td>
    <td>CUDA</td>
    <td>1</td>
    <td><b>29</b></td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 1 --preferred-device-type CUDA -t --debug</code></td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>CUDA</td>
    <td>4</td>
    <td>51</td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 4 --preferred-device-type CUDA --gpu-only -t --debug</code></td>
  </tr>
  <tr style="background: grey">
    <td>GPU</td>
    <td>CUDA</td>
    <td>1</td>
    <td><b>28</b></td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 1 --preferred-device-type CUDA --gpu-only -t --debug</code></td>
  </tr>

  <tr>
    <td>CPU</td>
    <td>CPU</td>
    <td>0</td>
    <td>485</td>
    <td><code>python3 render_blender_main.py /path/to/ShapeNetV2 -s ./test_speed --gpu-count 0 -t --debug</code></td>
  </tr>
  
</table>

From this table difference between CUDA and OPTIX is small, and also single GPU for single process is much faster.


Example of how use blender scripts on bunny can be found in [this notebook](./render_blender/test.ipynb).

Example of the render view on `bunny.obj` could be found [here](./render_blender/bunny_views).
Few examples of final images:
<table>
  <tr>
    <th>
      <center>Image</center>
    </th>
  </tr>
  <tr>
    <td><img src="./render_blender/bunny_views/rendering/00.png"></td>
  </tr>
  <tr>
    <td><img src="./render_blender/bunny_views/rendering/01.png"></td>
  </tr>
  <tr>
    <td><img src="./render_blender/bunny_views/rendering/02.png"></td>
  </tr>
  <tr>
    <td><img src="./render_blender/bunny_views/rendering/03.png"></td>
  </tr>
</table>

Depth are stored in `.exp` files as [`OpenEXR`](https://openexr.com/en/latest/) format. To open files fast and without any other depedency [py-minexr](https://github.com/cheind/py-minexr/tree/develop) is used. How to read saved depth cound found in [here](./render_blender/test.ipynb). Its been tested that [official OpenEXR](https://github.com/sanguinariojoe/pip-openexr) and this py-minexr libs produce similar results. Keep in mind that py-minexr produce float32, while official OpenEXR module could produce float64.

Known problems:
- Not well tested code for now. There could be some bugs with render.
- Object files from ShapeNetV1 doesn't loaded properly by Blender with version 3.5.0. If we attempt to load it will be dropped "Segmentation fault". ShapeNetV2 works as expected and its tested.
- Its seems that after sometime certain objects are skipped in render pipeline. Need to investigate this. For now temporary solution is to start scripts again where by default will be skiped models with created views.


## Point sample
The main purpose of this module is to generate points from big-size voxel (256<sup>3</sup>). Most of the code taken from [IM-NET dataset preparation](https://github.com/czq142857/IM-NET/tree/master/point_sampling). But personally I found this code very slow and in order to speed up it - I use Numba here, which gives a huge boost compare to original code.

How to use code in the folder `point_sample` on ShapeNet dataset:
```bash
python3 point_sample_main.py /path/to/ShapeNetV2 \
  -s /path/to/save/gen-points -n 6
```

If you want to use points along with renders, you can pass folder path to the folder with renders to add them into final h5 file. Example command:
```bash
python3 point_sample_main.py /path/to/ShapeNetV2 \
  -s /path/to/save/gen-points -n 6 -r /path/to/ShapeNetRenderingsV2
```

Example of how use point sample scripts on bunny can be found in [this notebook](./point_sample/test.ipynb).


## H5File dataset creation
After voxelization, render and point sample stages are ready, final data (from point-sample) stored as separate h5files, but its easy to work with file where all classes (objects) stored in single h5file which is suitable for further training or exploring dataset. This can be done in utils folder.

Next command (in utils folder), will combine h5files into single one:
```bash
python3 combine_h5_files.py /path/to/saved/h5files \
  --create-indx2model-id-file -s /path/to/saved/h5files --width 127 --height 127
```

Next command, will combine h5files into two files for training and testing:
```bash
python3 combine_h5_files.py /path/to/saved/h5files \
  --create-indx2model-id-file -s /path/to/saved/h5files --width 127 --height 127 --split --perc-train
```
Notice that size (Width and Height) must be the same for every file from render stage.

if you want to have certain models ids to train or test sets. You need to first create single one big h5 file, and then run:
```bash
python3 split_h5_file_to_train_test.py /path/to/saved/h5file/dataset.hdf5 \
  --with-renders --width 127 --height 127 --txt-train-ids-filename-path /path/to/txt/with/ids/train_model_ids.txt
```

More arguments and their description could be found via help.

# License
This project is licensed under the terms of the MIT license (see LICENSE for details).
