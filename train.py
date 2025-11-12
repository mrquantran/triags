import os
from scene.app_model import AppModel
import torch
import random
import numpy as np
import cv2
from utils.loss_utils import l1_loss, lncc, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import torch.nn.functional as F
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import point_double_to_normal, depth_double_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from scene.cameras import Camera

# Geman-McClure loss function (special case of Barron's robust loss with alpha = -2)
# This loss is robust to outliers and is often used for regression tasks.
# The formula is:
#     loss = (error^2 / 2) / ( (error^2 / 2) + scale^2 )
# - 'error' is the difference between prediction and target.
# - 'scale' controls how strongly outliers are down-weighted (higher = more tolerant).
# This loss smoothly limits the influence of large errors, making optimization more stable.
def geman_mcclure_loss(error, scale = 0.1):
    error_sq = error**2
    return (error_sq / 2.0) / (error_sq / 2.0 + scale**2)

def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid

# function L1_loss_appearance is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py
def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]

    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]

    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    app_model = AppModel()
    app_model.train()
    app_model.cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_mask_loss_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    ema_tgpc_loss_for_log = 0.0
    geo_loss, ncc_loss, tgpc_loss = None, None, None

    require_depth = not dataset.use_coord_map
    require_coord = dataset.use_coord_map

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, kernel_size, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        reg_kick_on = iteration >= opt.regularization_from_iter

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size, require_coord = require_coord and reg_kick_on, require_depth = require_depth and reg_kick_on, app_model=app_model)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
                                                                    render_pkg["render"],
                                                                    render_pkg["viewspace_points"],
                                                                    render_pkg["visibility_filter"],
                                                                    render_pkg["radii"])
        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        if dataset.use_decoupled_appearance:
            Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1_render = l1_loss(rendered_image, gt_image)

        if reg_kick_on:
            lambda_depth_normal = opt.lambda_depth_normal
            if require_depth:
                rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
                rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = depth_double_to_normal(viewpoint_cam, rendered_expected_depth, rendered_median_depth)
            else:
                rendered_expected_coord: torch.Tensor = render_pkg["expected_coord"]
                rendered_median_coord: torch.Tensor = render_pkg["median_coord"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = point_double_to_normal(viewpoint_cam, rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
        else:
            lambda_depth_normal = 0
            depth_normal_loss = torch.tensor([0],dtype=torch.float32,device="cuda")

        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image.unsqueeze(0)))

        loss = rgb_loss + depth_normal_loss * lambda_depth_normal

        # multi-view loss
        if iteration > opt.regularization_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False

            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                tgpc_weight = getattr(opt, 'tgpc_loss_weight', 0.1)
                tgpc_num_neighbors = getattr(opt, 'tgpc_num_neighbors', 1)

                ## compute geometry consistency mask and loss
                H, W = render_pkg['median_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['median_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, background, kernel_size, require_coord = require_coord and reg_kick_on, require_depth = require_depth and reg_kick_on, app_model=app_model)
                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['median_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['median_depth'], pts_in_nearest_cam)

                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    img_show = ((rendered_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((rendered_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)

                    depth = render_pkg['median_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

                    depth_expected = render_pkg['expected_depth'].squeeze().detach().cpu().numpy()
                    depth_expected_i = (depth_expected - depth_expected.min()) / (depth_expected.max() - depth_expected.min() + 1e-20)
                    depth_expected_i = (depth_expected_i * 255).clip(0, 255).astype(np.uint8)
                    depth_expected_show = cv2.applyColorMap(depth_expected_i, cv2.COLORMAP_JET)

                    row0 = np.concatenate([gt_img_show, img_show, normal_show], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_expected_show], axis=1)

                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss

                    if tgpc_weight > 0.0 and d_mask.sum() > 0 :
                        final_d_mask_pgsr_flat = d_mask.reshape(-1)
                        candidate_indices_for_tgpc = torch.arange(final_d_mask_pgsr_flat.shape[0], device=final_d_mask_pgsr_flat.device)[final_d_mask_pgsr_flat]

                        final_valid_indices_for_tgpc = candidate_indices_for_tgpc

                        if iteration % 2500 == 0: print("TGPC sampling num_valid_indices: ", final_valid_indices_for_tgpc.numel())
                        if final_valid_indices_for_tgpc.numel() > 0:
                            # Get 3D world points X_r for these sampled reference pixels
                            X_r_sampled_tgpc = pts.reshape(-1,3)[final_valid_indices_for_tgpc]

                            # Get 2D pixel coordinates p_r for these sampled points in the reference view
                            p_r_sampled_coords_tgpc = pixels.reshape(-1,2)[final_valid_indices_for_tgpc]
                            u_r_tgpc, v_r_tgpc = p_r_sampled_coords_tgpc[:,0], p_r_sampled_coords_tgpc[:,1]

                            # Get projection matrix P_r for reference view (viewpoint_cam)
                            K_r_tgpc = viewpoint_cam.get_k() # Intrinsic matrix K_r
                            RT_r_w2c_tgpc = viewpoint_cam.world_view_transform.transpose(0,1)[:3,:] # Extrinsic [R_wc | T_wc]
                            P_r_tgpc = K_r_tgpc @ RT_r_w2c_tgpc # P_r = K_r [R_r | T_r] (using world-to-cam extrinsics)

                            # List to store rows of the triangulation matrix A_Xr
                            A_rows_list_tgpc = []

                            row0_r = u_r_tgpc.unsqueeze(-1) * P_r_tgpc[2,:] - P_r_tgpc[0,:]
                            row1_r = v_r_tgpc.unsqueeze(-1) * P_r_tgpc[2,:] - P_r_tgpc[1,:]

                            # Add equations from the reference view (viewpoint_cam)
                            Ac_r = torch.stack([row0_r, row1_r], dim=1)
                            norm_Ac_r = torch.linalg.norm(Ac_r, ord='fro', dim=(1,2), keepdim=True)
                            Ac_r_normalized = (Ac_r / norm_Ac_r)
                            A_rows_list_tgpc.append(Ac_r_normalized[:,0,:].unsqueeze(1))
                            A_rows_list_tgpc.append(Ac_r_normalized[:,1,:].unsqueeze(1))

                            # --- Select up to 'tgpc_num_neighbors' neighbor cameras for TGPC ---
                            selected_neighbor_cams_for_tgpc = []
                            neighbor_ids = viewpoint_cam.nearest_id
                            if neighbor_ids:
                                num_to_sample = min(tgpc_num_neighbors, len(neighbor_ids))
                                chosen_indices = random.sample(neighbor_ids, num_to_sample)
                                selected_neighbor_cams_for_tgpc = [scene.getTrainCameras()[idx] for idx in chosen_indices]

                            # Ensure nearest_cam is included if needed
                            if not use_virtul_cam and nearest_cam is not None and nearest_cam not in selected_neighbor_cams_for_tgpc:
                                selected_neighbor_cams_for_tgpc.append(nearest_cam)
                                selected_neighbor_cams_for_tgpc = selected_neighbor_cams_for_tgpc[:tgpc_num_neighbors]

                            for neighbor_cam_for_tgpc in selected_neighbor_cams_for_tgpc:
                                X_r_in_current_neighbor_space = X_r_sampled_tgpc @ neighbor_cam_for_tgpc.world_view_transform[:3,:3] + neighbor_cam_for_tgpc.world_view_transform[3,:3]
                                z_curr_n_proj_tgpc = X_r_in_current_neighbor_space[:, 2:3]
                                p_curr_n_sampled_coords_tgpc = torch.stack(
                                    [X_r_in_current_neighbor_space[:,0] * neighbor_cam_for_tgpc.Fx / z_curr_n_proj_tgpc.squeeze(-1) + neighbor_cam_for_tgpc.Cx,
                                     X_r_in_current_neighbor_space[:,1] * neighbor_cam_for_tgpc.Fy / z_curr_n_proj_tgpc.squeeze(-1) + neighbor_cam_for_tgpc.Cy], dim=-1)
                                u_curr_n_tgpc, v_curr_n_tgpc = p_curr_n_sampled_coords_tgpc[:,0], p_curr_n_sampled_coords_tgpc[:,1]

                                K_curr_n_tgpc = neighbor_cam_for_tgpc.get_k()
                                RT_curr_n_w2c_tgpc = neighbor_cam_for_tgpc.world_view_transform.transpose(0,1)[:3,:]
                                P_curr_n_tgpc = K_curr_n_tgpc @ RT_curr_n_w2c_tgpc

                                row0_n_curr = u_curr_n_tgpc.unsqueeze(-1) * P_curr_n_tgpc[2,:] - P_curr_n_tgpc[0,:]
                                row1_n_curr = v_curr_n_tgpc.unsqueeze(-1) * P_curr_n_tgpc[2,:] - P_curr_n_tgpc[1,:]
                                Ac_n_curr = torch.stack([row0_n_curr, row1_n_curr], dim=1)
                                norm_Ac_n_curr = torch.linalg.norm(Ac_n_curr, ord='fro', dim=(1,2), keepdim=True)
                                Ac_n_curr_normalized = Ac_n_curr / norm_Ac_n_curr
                                A_rows_list_tgpc.append(Ac_n_curr_normalized[:,0,:].unsqueeze(1))
                                A_rows_list_tgpc.append(Ac_n_curr_normalized[:,1,:].unsqueeze(1))

                            assert (2 * len(selected_neighbor_cams_for_tgpc) + 2) == len(A_rows_list_tgpc) # 2(k + 1)
                            assert len(A_rows_list_tgpc) >= 4, "Not enough equations for triangulation in TGPC loss computation"

                            A_Xr_tgpc = torch.cat(A_rows_list_tgpc, dim=1)

                            try:
                                # _U_svd: (N_sampled, N_eqs, min(N_eqs, 4))
                                # S_svd: (N_sampled, min(N_eqs, 4))
                                # Vh_svd: (N_sampled, 4, 4) (V transpose)
                                _U_svd, S_svd, Vh_svd = torch.linalg.svd(A_Xr_tgpc, full_matrices=False)

                                # X_triangulated_homo is the last column of V (last row of Vh)
                                X_triangulated_homo_tgpc = Vh_svd[:, -1, :] # Shape: (N_sampled, 4)

                                # Convert homogeneous X_triangulated to Cartesian 3D - Ensure the homogeneous coordinate is positive before dividing
                                X_triangulated_homo_tgpc = X_triangulated_homo_tgpc / (X_triangulated_homo_tgpc[:, 3:4].clone().abs().clamp(min=1e-8) * torch.sign(X_triangulated_homo_tgpc[:, 3:4].clone().clamp(min=1e-8))) # Avoid div by zero or negative W
                                X_triangulated_cartesian_tgpc = X_triangulated_homo_tgpc[:, :3] # Shape: (N_sampled, 3)

                                # ------------------------------------------| Depth Distance Wegihts
                                ref_depths_sampled = render_pkg['median_depth'].squeeze().reshape(-1)[final_valid_indices_for_tgpc]
                                inverse_depths = (1.0 / ref_depths_sampled)
                                depth_weights = inverse_depths / torch.max(inverse_depths).detach()
                                depth_weights = depth_weights.detach()

                                # ------------------------------------------| GEMAN-MCCLURE LOSS
                                scale = 0.1
                                error_3d = X_r_sampled_tgpc - X_triangulated_cartesian_tgpc
                                gm_loss_per_dimension = geman_mcclure_loss(error_3d, scale=scale) # Shape: (N_sampled, 3)
                                threeD_distance = gm_loss_per_dimension.sum(dim=1)

                                # ------------------------------------------| Reprojection weights
                                weights_for_tgpc = weights.reshape(-1)[final_valid_indices_for_tgpc]

                                # Final TGPC loss calculation
                                weights_final = depth_weights * weights_for_tgpc
                                current_tgpc_loss = (weights_final * threeD_distance).mean()
                                loss += tgpc_weight * current_tgpc_loss
                                tgpc_loss = current_tgpc_loss # @TODO: logging
                            except torch.linalg.LinAlgError:
                                assert False, "SVD failed during TGPC loss computation"

                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ## Render Distance
                        rays_d = viewpoint_cam.get_rays()
                        rendered_normal2 = render_pkg["normal"].permute(1,2,0).reshape(-1,3)
                        ref_local_d = render_pkg['median_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        ref_local_d = ref_local_d.reshape(*render_pkg['median_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1),
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            ema_tgpc_loss_for_log = 0.4 * tgpc_loss.item() if tgpc_loss is not None else 0.0 + 0.6 * ema_tgpc_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{4}f}",
                    "loss_normal": f"{ema_normal_loss_for_log:.{4}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{8}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{8}f}",
                    "TGPC": f"{ema_tgpc_loss_for_log:.{8}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1_render, loss, depth_normal_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, kernel_size), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)

    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, normal_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if config["name"] == "test":
                    with open(scene.model_path + "/chkpnt" + str(iteration) + ".txt", "w") as file_object:
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), file=file_object)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset=lp.extract(args),
             opt=op.extract(args),
             pipe=pp.extract(args),
             testing_iterations=args.test_iterations,
             saving_iterations=args.save_iterations,
             checkpoint_iterations=args.checkpoint_iterations,
             checkpoint=args.start_checkpoint,
             debug_from=args.debug_from)

    # All done
    print("\nTraining complete.")
