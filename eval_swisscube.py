from train_swisscube import *
MINIBATCH_SIZE = 4


def evaluate(gen_samples, network):
    
    global weights_rot, extents, points
    total_batches = 30356 // MINIBATCH_SIZE
    total_pose, total_flow = 0.0, 0.0
    start = time.time()
    
    for curr_batch, sample in enumerate(gen_samples):
        images, flows, poses_src, poses_tgt, affine_matrices, zoom = sample

        # zoom in image
        grids = nn.functional.affine_grid(affine_matrices, images.size(), align_corners=False)
        input_zoom = nn.functional.grid_sample(images, grids, align_corners=False).detach()

        # zoom in flow
        flow_zoom = nn.functional.grid_sample(flows, grids)
        for k in range(flow_zoom.shape[0]):
            flow_zoom[k, 0, :, :] /= affine_matrices[k, 0, 0] * 20.0
            flow_zoom[k, 1, :, :] /= affine_matrices[k, 1, 1] * 20.0

        output, loss_pose_tensor, quaternion_delta_var, translation_var = \
            network(input_zoom.float(), tweights_rot.float(),
                    torch.from_numpy(poses_src).cuda().detach(), torch.from_numpy(poses_tgt).cuda().detach(),
                    textents.float(), tpoints.float(), zoom.float())


        quaternion_delta = quaternion_delta_var.cpu().detach().numpy()
        translation = translation_var.cpu().detach().numpy()
        
        
        try:
            poses_est, error_rot, error_trans = \
                _compute_pose_target(quaternion_delta, translation, poses_src, poses_tgt)
        
            for i in images.shape[0]:
                img_real = images[i, :3].cpu().numpy() * 255
                img_real = np.transpose(img_real, (1, 2, 0))
                
                cfg.renderer.set_pose(poses_est[i])
                img_render = cfg.renderer.render_()
                img_seg = (img_render > 0).astype(np.float32)
                img_seg[:, :, :2] = 0

                both = cv2.addWeighted(img_real, 0.4, img_seg, 0.1, 0)
                cv2.imshow('diff', both)
                key = cv2.waitKey(0)

                if key == 27:
                    exit(0)
        except:
            error_rot = np.nan
            error_trans = np.nan

        

if __name__ == '__main__':
    cfg_from_file('experiments/cfgs/swisscube.yml')

    cfg.renderer = Renderer(synthetic=True)
    network_path = 'data/checkpoints/swisscube/latest.pth'
    network = load_network(network_path)

    evaluate(generate_samples('validation'), network)
