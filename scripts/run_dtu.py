import os

scenes = [24]
data_base_path='data/dtu_dataset/dtu'
out_base_path='output_dtu/'
eval_path='data/dtu_dataset/dtu_eval'
out_name='test'
gpu_id=0

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    common_args = "--ncc_scale 0.5 -r 2 --multi_view_num 12 --regularization_from_iter 12000 --use_decoupled_appearance"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = ""
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    # common_args = " "
    # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python mesh_extract.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    # print(cmd)
    # os.system(cmd)

    common_args = f" --num_cluster 1 --voxel_size 0.002 --depth_trunc 20.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} timeout 5m python tsdf_fusion.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} timeout 8m python scripts/eval_dtu/evaluate_single_scene.py " + \
        f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/train/ours_30000/fuse_post.ply " + \
        f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/train/ours_30000 " + \
        f"--mask_dir {data_base_path} " + \
        f"--DTU {eval_path}"
    print(cmd)
    os.system(cmd)