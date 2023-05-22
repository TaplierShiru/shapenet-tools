import numpy as np
import os
import random



def find_all_files_with_exts(root, exts):
    files_with_exts_list = []
    for path, subdirs, files in os.walk(root):
        for fname in files:
            fpath = os.path.join(path, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                files_with_exts_list.append(fpath)
    return files_with_exts_list


'''
#do not use progressive sampling (center2x2x2 -> 4x4x4 -> 6x6x6 ->...)
#if sample non-center points only for inner(1)-voxels,
#the reconstructed model will have railing patterns.
#since all zero-points are centered at cells,
#the model will expand one-points to a one-planes.
'''
def sample_point_in_cube(block,target_value,halfie):
    halfie2 = halfie * 2
    
    for i in range(100):
        x = np.random.randint(halfie2)
        y = np.random.randint(halfie2)
        z = np.random.randint(halfie2)
        if block[x,y,z] == target_value:
            return x, y, z
    
    if block[halfie, halfie, halfie] == target_value:
        return halfie, halfie, halfie
    
    i = 1
    ind = np.unravel_index(
        np.argmax(block[halfie-i:halfie+i, halfie-i:halfie+i, halfie-i:halfie+i], axis=None), 
        (i*2, i*2, i*2)
    )
    if block[ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i] == target_value:
        return ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i
    
    for i in range(2,halfie+1):
        # If we assume that its 2d, when these points would be like a plus (or cross)
        six = [
            (halfie-i,   halfie,     halfie),
            (halfie+i-1, halfie,     halfie),
            (halfie,     halfie,     halfie-i),
            (halfie,     halfie,     halfie+i-1),
            (halfie,     halfie-i,   halfie),
            (halfie,     halfie+i-1, halfie)
        ]
        for coords in six:
            if block[coords] == target_value:
                return coords
        ind = np.unravel_index(
            np.argmax(block[halfie-i:halfie+i, halfie-i:halfie+i, halfie-i:halfie+i], axis=None), 
            (i*2, i*2, i*2)
        )
        if block[ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i] == target_value:
            return ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i
    raise Exception('hey, error in your code!')


def carve_voxels_alg(big_voxel, dim_voxel, top_view, left_min, left_max, front_min, front_max):
    for j in range(dim_voxel):
        for k in range(dim_voxel):
            occupied = False
            for i in range(dim_voxel):
                if big_voxel[i,j,k]>0:
                    if not occupied:
                        occupied = True
                        left_min[j,k] = i
                    left_max[j,k] = i

    for i in range(dim_voxel):
        for j in range(dim_voxel):
            occupied = False
            for k in range(dim_voxel):
                if big_voxel[i,j,k]>0:
                    if not occupied:
                        occupied = True
                        front_min[i,j] = k
                    front_max[i,j] = k

    for i in range(dim_voxel):
        for k in range(dim_voxel):
            if top_view[i,k]>0:
                fill_flag = False
                for j in range(dim_voxel-1,-1,-1):
                    if big_voxel[i,j,k]>0:
                        fill_flag = True
                    else:
                        if left_min[j,k]<i and left_max[j,k]>i and front_min[i,j]<k and front_max[i,j]>k:
                            if fill_flag:
                                big_voxel[i,j,k]=1
                        else:
                            fill_flag = False

    
def carve_voxels(big_voxel: np.ndarray):
    dim_voxel = big_voxel.shape[0]
    # Numba doesnt support max with axis, so prepara array before usage of alg
    top_view = np.max(big_voxel, axis=1)
    left_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
    left_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
    front_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
    front_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
    carve_voxels_alg(big_voxel, dim_voxel, top_view, left_min, left_max, front_min, front_max)


def compress_big_voxel(big_voxel: np.ndarray, dim_to_reduce: int):
    voxel_model_temp = np.zeros((dim_to_reduce,dim_to_reduce,dim_to_reduce),np.uint8)
    multiplier = int(big_voxel.shape[0]/dim_to_reduce)
    for i in range(dim_to_reduce):
        for j in range(dim_to_reduce):
            for k in range(dim_to_reduce):
                voxel_model_temp[i,j,k] = np.max(
                    big_voxel[
                        i*multiplier:(i+1)*multiplier,
                        j*multiplier:(j+1)*multiplier,
                        k*multiplier:(k+1)*multiplier
                    ]
                )
    return np.reshape(voxel_model_temp, (dim_to_reduce,dim_to_reduce,dim_to_reduce,1))


def sample_points_near_surface(big_voxel: np.ndarray, dim_to_reduce: int, points_to_sample: int):
    sample_voxels = compress_big_voxel(big_voxel, dim_to_reduce)
    sample_points = np.zeros((points_to_sample,3),np.uint8)
    sample_values = np.zeros((points_to_sample,1),np.uint8)
    
    multiplier = int(big_voxel.shape[0]/dim_to_reduce)
    halfie = int(multiplier/2)
    points_sampled = 0
    voxel_model_temp_flag = np.zeros((dim_to_reduce,dim_to_reduce,dim_to_reduce),np.uint8)
    temp_range = list(
        range(1, dim_to_reduce-1, 4)
    )+list(
        range(2, dim_to_reduce-1, 4)
    )+list(
        range(3, dim_to_reduce-1, 4)
    )+list(
        range(4, dim_to_reduce-1, 4)
    )
    for j in temp_range:
        if (points_sampled>=points_to_sample): break
        for i in temp_range:
            if (points_sampled>=points_to_sample): break
            for k in temp_range:
                if (points_sampled>=points_to_sample): break
                if (np.max(sample_voxels[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(sample_voxels[i-1:i+2,j-1:j+2,k-1:k+2])):
                    si,sj,sk = sample_point_in_cube(
                        big_voxel[
                            i*multiplier:(i+1)*multiplier,
                            j*multiplier:(j+1)*multiplier,
                            k*multiplier:(k+1)*multiplier
                        ],
                        sample_voxels[i,j,k],
                        halfie
                    )
                    sample_points[points_sampled,0] = si+i*multiplier
                    sample_points[points_sampled,1] = sj+j*multiplier
                    sample_points[points_sampled,2] = sk+k*multiplier
                    sample_values[points_sampled] = sample_voxels[i,j,k]
                    voxel_model_temp_flag[i,j,k] = 1
                    points_sampled +=1
    if (points_sampled>=points_to_sample):
        # print(f"{dim_to_reduce}-- batch_size exceeded!")
        exceed_flag = 1
    else:
        exceed_flag = 0
        #fill other slots with random points
        while (points_sampled<points_to_sample):
            while True:
                i = random.randint(0,dim_to_reduce-1)
                j = random.randint(0,dim_to_reduce-1)
                k = random.randint(0,dim_to_reduce-1)
                if voxel_model_temp_flag[i,j,k] != 1: break
            si,sj,sk = sample_point_in_cube(
                big_voxel[
                    i*multiplier:(i+1)*multiplier,
                    j*multiplier:(j+1)*multiplier,
                    k*multiplier:(k+1)*multiplier
                ],
                sample_voxels[i,j,k],
                halfie
            )
            sample_points[points_sampled,0] = si+i*multiplier
            sample_points[points_sampled,1] = sj+j*multiplier
            sample_points[points_sampled,2] = sk+k*multiplier
            sample_values[points_sampled] = sample_voxels[i,j,k]
            voxel_model_temp_flag[i,j,k] = 1
            points_sampled +=1
    return sample_voxels, sample_points, sample_values, exceed_flag