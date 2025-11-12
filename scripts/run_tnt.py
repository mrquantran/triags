import os

scenes = ['Truck', 'Train', 'Playground', 'Barn', 'Caterpillar', 'Family', 'Ignatius', 'Meetingroom', 'Church', 'Courtroom', 'Museum', 'Palace', 'Temple']
data_devices = ['cuda'] * len(scenes)
data_base_path = '/home/username/PGSR/data/tnt'
out_base_path='output_tnt_ablation_3'
out_name='test'
gpu_id=0

for id, scene in enumerate(scenes):
    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = f"--eval -r2 --ncc_scale 0.5 --data_device {data_devices[id]} --regularization_from_iter 12000 --use_decoupled_appearance --multi_view_num 12 --exposure_compensation"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} timeout 2h python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--eval --max_depth 20.0 --voxel_size 0.002 --data_device {data_devices[id]}"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} timeout 5m python scripts/render_tnt.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    # require open3d==0.9
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} timeout 25m python scripts/tnt_eval/run.py --dataset-dir {data_base_path}/{scene} --traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {out_base_path}/{scene}/{out_name}/mesh/tsdf_fusion_post.ply --out-dir {out_base_path}/{scene}/{out_name}/mesh'
    print(cmd)
    os.system(cmd)