from ycb_renderer import YCBRenderer
import os
import argparse
import numpy as np


if __name__ == "__main__":
    model_path = "/cvlabdata2/home/protopap/deepim/data/"
    width, height = 480, 640

    renderer = YCBRenderer(width=width, height=height, render_marker=True)
    models = ["000001"]
    colors = [[0.9, 0, 0]]
    obj_paths = [
        '{}/models/{}/textured_simple.obj'.format(model_path, item) for item in models]
    texture_paths = [
        '{}/models/{}/texture_map.png'.format(model_path, item) for item in models]

    print(obj_paths)
    renderer.load_objects(obj_paths, texture_paths, colors)
    pose = np.array([-0.025801208, 0.08432201, 0.004528991,
                     0.9992879, -0.0021458883, 0.0304758, 0.022142926])

    theta = 0
    z = 1
    fix_pos = [np.sin(theta), z, np.cos(theta)]
    renderer.set_camera(fix_pos, [0, 0, 0], [0, 1, 0])
    fix_pos = np.zeros(3)
    renderer.set_poses([pose])
    cls_indexes = [0]
    renderer.set_light_pos([1, 1, 1])
    renderer.set_light_color([1., 1., 1.])
    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

    while True:
        renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)
        frame = [image_tensor.cpu().numpy(), seg_tensor.cpu().numpy()]
        centers = renderer.get_centers()
        for center in centers:
            x = int(center[1] * width)
            y = int(center[0] * height)
            frame[0][y-2:y+2, x-2:x+2, :] = 1
            frame[1][y-2:y+2, x-2:x+2, :] = 1
        if len(sys.argv) > 2 and sys.argv[2] == 'headless':
            # print(np.mean(frame[0]))
            theta += 0.001
            if theta > 1:
                break
        else:
            #import matplotlib.pyplot as plt
            #plt.imshow(np.concatenate(frame, axis=1))
            # plt.show()
            cv2.imshow('test', cv2.cvtColor(
                np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
            q = cv2.waitKey(16)
            if q == ord('w'):
                z += 0.05
            elif q == ord('s'):
                z -= 0.05
            elif q == ord('a'):
                theta -= 0.1
            elif q == ord('d'):
                theta += 0.1
            elif q == ord('p'):
                Image.fromarray(
                    (frame[0][:, :, :3] * 255).astype(np.uint8)).save('test.png')
            elif q == ord('q'):
                break
            elif q == ord('r'):  # rotate
                pose[3:] = qmult(axangle2quat(
                    [0, 0, 1], 5/180.0 * np.pi), pose[3:])
                pose2[3:] = qmult(axangle2quat(
                    [0, 0, 1], 5 / 180.0 * np.pi), pose2[3:])
                pose3[3:] = qmult(axangle2quat(
                    [0, 0, 1], 5 / 180.0 * np.pi), pose3[3:])
                renderer.set_poses([pose, pose2, pose3])

        cam_pos = fix_pos + np.array([np.sin(theta), z, np.cos(theta)])
        renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
        #renderer.set_light_pos(cam_pos)
    
    dt = time.time() - start
    print("{} fps".format(1000 / dt))

    renderer.release()
