import numpy as np
import pickle
import json
import os
from tqdm import tqdm

def solve_translation(X, x, K):
    A = np.zeros((2*X.shape[0], 3))
    b = np.zeros((2*X.shape[0], 1))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    for nj in range(X.shape[0]):
        A[2*nj, 0] = 1
        A[2*nj + 1, 1] = 1
        A[2*nj, 2] = -(x[nj, 0] - cx)/fx
        A[2*nj+1, 2] = -(x[nj, 1] - cy)/fy
        b[2*nj, 0] = X[nj, 2]*(x[nj, 0] - cx)/fx - X[nj, 0]
        b[2*nj+1, 0] = X[nj, 2]*(x[nj, 1] - cy)/fy - X[nj, 1]
        A[2*nj:2*nj+2, :] *= x[nj, 2]
        b[2*nj:2*nj+2, :] *= x[nj, 2]
    trans = np.linalg.inv(A.T @ A) @ A.T @ b
    return trans.T[0]

class InitTranslation:
    def __init__(self, solve_T=True, solve_R=False) -> None:
        self.solve_T = solve_T
        self.solve_R = solve_R
    
    def __call__(self, kpts1, params, cameras, keypoints):
        """ Refine SMPL parameters using PnP

        Args:
            body_model (SMPLModel): The SMPL model instance.
            params (list): List of dictionaries containing SMPL parameters, with shape (#nframes, 1).
                Keys: ['Rh', 'Th', 'poses', 'shapes', 'inv_trans']  # only 'Th' and/or 'Rh' are used here 
            cameras (list): List of dictionaries containing camera parameters, with shape (#nframes, 1).
                Keys: ['K', 'R', 'T', 'dist']   # only 'K' is used here
            keypoints (numpy.ndarray): Array of shape (#frames, 1, 25, 3) containing 2D keypoints.
        """
        nJoints = 15    # Use the 15 principal SMPL joints
        params['Th'] = np.zeros_like(params['Th'])
        #kpts1 = body_model.keypoints(params, return_tensor=False)
        for i in range(kpts1.shape[0]):
            k2d = keypoints[i, :nJoints]
            if k2d[:, -1].sum() < nJoints / 2:
                print('[{}] No valid keypoints in frame {}'.format(self.__class__.__name__, i))
                params['Th'][i] = params['Th'][i-1]
                continue
            trans = solve_translation(kpts1[i, :nJoints], k2d, cameras['K'][i])
            params['Th'][i] += trans
        # params['shapes'] = params['shapes'].mean(0, keepdims=True)
        return {'params': params}
    
    
if __name__ == "__main__":
    # smpl_loader = SMPLLoader(
    #     'data/smpl-meta/SMPL_NEUTRAL.pkl',
    #     'regressor_path: data/smpl-meta/J_regressor_body25.npy')
    # body_model = smpl_loader.smplmodel
    
    path = "/root/gauhuman/GauHuman/data/andy2"    # data root
    poses_path = f'{path}/poses_optimized.npz'
    # Read init parameters
    # states_path = '/home/haodongw/workspace/EasyMocap_new/output/rotation_landscape/states.pkl'
    # with open(states_path, 'rb') as file:
    #     loaded_args = pickle.load(file)
    
    body_pose = np.load(poses_path)['body_pose']
    betas = np.load(poses_path)['betas']
    global_orient = np.load(poses_path)['global_orient']
    transl = np.load(poses_path)['transl']
    intrinsic = np.load(f'{path}/cameras.npz')['intrinsic']
    frames = global_orient.shape[0]
    keypoints = np.load(f'{path}/keypoints.npy')
    #keypoints = np.load(f'{path}/keypoints.npy').reshape((frames, 1, 25, 3))
    # in the wild

    H, W = 1280, 720
    coeff = 1.2
    f = coeff * min(W, H)
    K = np.array([[f, 0, W//2], [0, f, H//2],[0, 0, 1]])


    params = {'Rh': global_orient, 'Th': transl, 'poses': body_pose, 'shapes': np.full((frames, len(betas)), betas)}
    cameras = {'K': [K] * frames}

    #body_model = smpl_numpy.SMPL('neutral', 'data/smpl-meta')
    #body_model = SMPL(model_path="data/smpl-meta/SMPL_NEUTRAL.pkl")
    #a, b = body_model()
    kpts1 = np.load(f"{path}/verts.npy")
    
    
    # Refinement
    ref = InitTranslation(solve_T=True, solve_R=False)
    res = ref(kpts1, params, cameras, keypoints)
    # Save results
    save_path = f"{path}/smpl_refined"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    breakpoint() 
    max_length = max(len(value) for key, value in res['params'].items() if isinstance(value, np.ndarray))
    for i in tqdm(range(max_length)):
        index_data = {}
        for key, value in res['params'].items():
            if isinstance(value, np.ndarray):
                index_data[key] = value[i].reshape(1, -1).tolist() 
        file_name = f'{i:06d}.json'  # Naming files as params_index.json
        full_file_path = os.path.join(save_path, file_name)  # Construct full file path
        with open(full_file_path, 'w') as file:
            wrapped_index_data = [index_data]
            json.dump(wrapped_index_data, file, indent=4)  # Dump the index data to the file
