import numpy as np
import random
import numba as nb

from point_sample_tools import carve_voxels_alg as source_carve_voxels_alg, compress_big_voxel as source_compress_big_voxel


carve_voxels_alg = nb.njit()(source_carve_voxels_alg)
compress_big_voxel = nb.njit()(source_compress_big_voxel)


@nb.njit()
def find_index_first_maximum(data):
    """
    This function is re-implementation of operation:
        np.unravel_index(np.argmax(data, axis=None), data.shape)
    Which gives index of max (and first) value, but numba does not support it
    So in order to speed up preparation, I make it simple as here
    
    """
    assert len(data.shape) == 3
    max_value = data[0, 0, 0]
    max_value_index = (0, 0, 0)
    
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            for kk in range(data.shape[2]):
                if data[ii, jj, kk] > max_value:
                    max_value = max_value
                    max_value_index = (ii, jj, kk)
    return max_value, max_value_index


@nb.njit()
def sample_point_in_cube(block, target_value, halfie):
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
    _, ind = find_index_first_maximum(
        block[halfie-i:halfie+i, halfie-i:halfie+i, halfie-i:halfie+i]
    )
    if block[ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i] == target_value:
        return ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i
    
    for i in range(2, halfie + 1):
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
        _, ind = find_index_first_maximum(
            block[halfie-i:halfie+i, halfie-i:halfie+i, halfie-i:halfie+i]
        )
        if block[ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i] == target_value:
            return ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i
    raise Exception('hey, error in your code!')

                            
def carve_voxels(big_voxel: np.ndarray):
    dim_voxel = big_voxel.shape[0]
    # Numba doesnt support max with axis, so prepara array before usage of alg
    top_view = np.max(big_voxel, axis=1)
    left_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
    left_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
    front_min = np.full([dim_voxel,dim_voxel],dim_voxel,np.int32)
    front_max = np.full([dim_voxel,dim_voxel],-1,np.int32)
    carve_voxels_alg(big_voxel, dim_voxel, top_view, left_min, left_max, front_min, front_max)


@nb.njit()
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

