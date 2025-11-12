import os

scenes = ["drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
data_base_path='nerf_synthetic'
out_base_path = 'output_nerf_synthetic'
out_name='test'
gpu_id=0

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'cp -rf {data_base_path}/{scene}/sparse/0/* {data_base_path}/{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    common_args = "--eval --multi_view_num 12 --regularization_from_iter 15000"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} timeout 1h python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = "--skip_test --quiet"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} timeout 30m python render.py -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f" --voxel_size 0.002 --depth_trunc 20.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} timeout 10m python tsdf_fusion.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"timeout 2h python ./scripts/eval_nerf/eval_nerf.py " \
        f"--scene {scene} " \
        f"--mesh_path {out_base_path}/{scene}/{out_name}/train/ours_30000/fuse_post.ply " \
        f"--data_dir {data_base_path}/blender_eval/nerf_synthetic " \
        f"--evaluate " \
        f"--mask_filter"
    print(cmd)
    os.system(cmd)