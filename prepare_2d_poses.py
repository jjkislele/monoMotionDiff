import numpy as np

from utils.camera import normalize_screen_coordinates


def store_2d_norm_gt():
    h36m2d = np.load('data/data_2d_h36m_gt_gmm.npz', allow_pickle=True)['positions_2d'].tolist()
    h36m2d_norm = {}
    for scene_name, sV in h36m2d.items():
        print(scene_name + "===========")
        for action_name, mV in sV.items():
            # init
            if scene_name not in h36m2d_norm:
                h36m2d_norm[scene_name] = {}
            #######################
            joint2d = np.array(mV)[:, :, :, 0, 1:3]
            # calc uv
            joint2d = normalize_screen_coordinates(joint2d, 1000, 1000)
            h36m2d_norm[scene_name][action_name] = joint2d
            print(action_name)
    # store
    np.save('data/data_2d_h36m_gt_gmm_norm.npy', h36m2d_norm)


def store_2d_norm_cpn():
    h36m2d = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb_gmm.npz', allow_pickle=True)['positions_2d'].tolist()
    h36m2d_norm = {}
    gt = np.load('data/data_2d_h36m_gt_gmm_norm.npy', allow_pickle=True).tolist()
    for scene_name, sV in h36m2d.items():
        print(scene_name + "===========")
        for action_name, mV in sV.items():
            # init
            if scene_name not in h36m2d_norm:
                h36m2d_norm[scene_name] = {}
            #######################
            # trim
            gt_subject = gt[scene_name][action_name]
            try:
                joint2d = np.array(mV)[:, :, :, 0, 1:3]
            except:
                min_len = min([len(item) for item in mV])
                joint2d_ = [item[:min_len] for item in mV]
                joint2d = np.array(joint2d_)[:, :, :, 0, 1:3]
            if gt_subject.shape[1] != joint2d.shape[1]:
                assert gt_subject.shape[1] < joint2d.shape[1]
                joint2d = joint2d[:, :gt_subject.shape[1]]
            # calc uv
            joint2d = normalize_screen_coordinates(joint2d, 1000, 1000)
            h36m2d_norm[scene_name][action_name] = joint2d
            print(action_name)
    # store
    np.save('data/data_2d_h36m_cpn_ft_h36m_dbb_gmm_norm.npy', h36m2d_norm)


if __name__ == '__main__':
    store_2d_norm_gt()
    store_2d_norm_cpn()
