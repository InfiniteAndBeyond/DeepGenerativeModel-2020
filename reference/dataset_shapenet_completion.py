import torch.utils.data as data
import torch
import h5py
import numpy as np
import os
from glob import glob


class PartDataset_Completion(data.Dataset):
    def __init__(self, root = '/home/xiaoyuan/pointcloudGenerating/Completion/completion3d/tensorflow/data/shapenet',
                 npoints = 2048, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category_onlyPlane.txt')
        self.cat = {}

        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            # dir_point = os.path.join(self.root, self.cat[item], 'points')
            # dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            if train:
                split = "train"
            else:
                split = "val"
            dir_point_gt = os.path.join(self.root,split,'gt',self.cat[item])
            dir_point_partial = os.path.join(self.root,split,'partial',self.cat[item])
            #print(dir_point_gt, dir_point_partial)
            fns = sorted(os.listdir(dir_point_gt))

            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point_gt, token + '.h5'), os.path.join(dir_point_partial, token + '.h5')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        # self.num_seg_classes = 0
        # if not self.classification:
        #     for i in range(len(self.datapath)//50):
        #         l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
        #         if l > self.num_seg_classes:
        #             self.num_seg_classes = l
        # print("self.num_seg_classes: ",self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_gt = load_h5(fn[1])
        point_partial = load_h5(fn[2])
        #print(point_gt.shape, point_gt.shape)

        if point_gt.shape[0] != self.npoints:
            choice = np.random.choice(point_gt.shape[0], self.npoints, replace=True)
            #resample
            point_gt = point_gt[choice, :]
            point_partial = point_partial[choice, :]

        point_gt = torch.from_numpy(point_gt)
        point_partial = torch.from_numpy(point_partial)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_partial, point_gt

    def __len__(self):
        return len(self.datapath)


def load_h5(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    f = h5py.File(path, 'r')
    cloud_data = np.array(f['data'])
    f.close()
    return cloud_data.astype(np.float32)


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """Create a rotation matrix with an optional fourth homogeneous coordinate
    :param a, b, c: ZYZ-Euler angles
    """
    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, -np.sin(a), 0],
                         [0, 1, 0, 0],
                         [np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, False)
    return rot


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def atomic_rotate(data, angles):
    '''

    :param data: numpy array of Nx3 array
    :param angles: numpy array / list of 3
    :return: rotated_data: numpy array of Nx3
    '''
    R = angles2rotation_matrix(angles)
    rotated_data = np.dot(data, R)

    return rotated_data


def rotate_point_cloud_up(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    y_angle = np.random.uniform() * 2 * np.pi
    angles = [0, y_angle, 0]
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


def rotate_point_cloud_3d(data):
    # uniform sampling
    angles = np.random.rand(3) * np.pi * 2
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


if __name__ == '__main__':

    train_dataset = PartDataset_Completion(classification=True, npoints=2048)
    print(train_dataset.__getitem__(0))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    print(len(train_dataset))
