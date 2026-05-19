import glob
import re
import os
import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import random
class MMMP(BaseImageDataset):
    dataset_dir = '/data0/hewenke/mp-reid'

    def __init__(self,root = '' , verbose = True, pid_begin = 0,exp_setting = None, **kwargs):
        super(MMMP,self).__init__()
        # self.cfg = cfg
        self.dataset_dir = root if root else self.dataset_dir
        if exp_setting is None:
            raise ValueError("MMMP requires DATASETS.EXP_SETTING, e.g. 'exp_cctv_ir_cctv_rgb'.")
        self.exp_setting = exp_setting #
        self.setting_name_split = exp_setting.split("_")
        self.file_path_train = osp.join(self.dataset_dir,self.exp_setting,'train_id.txt')
        self.file_path_val   = osp.join(self.dataset_dir,self.exp_setting,'val_id.txt')
        self.file_path = osp.join(self.dataset_dir,self.exp_setting,'test_id.txt')
        self.pid_begin = pid_begin
        self.split_ratio = 0.5
        if len(self.setting_name_split) == 2: # exp_rgb
            train = self._process_train(self.dataset_dir,self.file_path_train,self.file_path_val,self.exp_setting, relabel=True)
            query, gallery = self._process_same(self.dataset_dir,self.file_path,self.exp_setting, relabel=False, split_ratio=self.split_ratio)
        elif len(self.setting_name_split) == 5: # exp_cctv_ir_cctv_rgb
            train = self._process_train(self.dataset_dir,self.file_path_train,self.file_path_val,self.exp_setting, relabel=True)
            query = self._process_query(self.dataset_dir,self.file_path,self.exp_setting, relabel=False)
            gallery = self._process_gallery(self.dataset_dir,self.file_path,self.exp_setting, relabel=False)
        else:
            raise ValueError(f"Unsupported MMMP exp_setting: {self.exp_setting}")

        if verbose:
            print("=> MMMP loaded")
            self.print_dataset_statistics(train, query, gallery)
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, camid))
        return dataset
    

    def _process_train(self, dir_path, file_path_train, file_path_val, exp_setting=None, relabel=True):
        """
        1. 读 train ID 文件
        2. 读 val ID 文件
        3. 把 ID 统一格式化成 '0001' 这种形式
        4. 根据 exp_setting 决定要用哪些 camera
        5. 遍历所有 pid 和 camera,对应目录里找图片
        6. 保存成 samples = [(img_path, pid, camid), ...]
        7. 把原始 pid 重映射成连续标签
        8. 生成 dataset = [(img_path, pid, camid, viewid), ...] 并返回
        """
        with open(file_path_train, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_train = ["%04d" % x for x in ids]

        with open(file_path_val, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_val = ["%04d" % x for x in ids]

        setting_name_split = exp_setting.split("_")
        train_cameras = []
        if len(setting_name_split) == 5:
            if setting_name_split[1] == 'cctv':
                if setting_name_split[2] == 'ir':
                    train_cameras += ['07', '08', '09', '10', '11', '12']
                elif setting_name_split[2] == 'rgb':
                    train_cameras += ['01', '02', '03', '04', '05', '06']
            elif setting_name_split[1] == 'uav':
                if setting_name_split[2] == 'ir':
                    train_cameras += ['14']
                elif setting_name_split[2] == 'rgb':
                    train_cameras += ['13']

            if setting_name_split[3] == 'cctv':
                if setting_name_split[4] == 'ir':
                    train_cameras += ['07', '08', '09', '10', '11', '12']
                elif setting_name_split[4] == 'rgb':
                    train_cameras += ['01', '02', '03', '04', '05', '06']
            elif setting_name_split[3] == 'uav':
                if setting_name_split[4] == 'ir':
                    train_cameras += ['14']
                elif setting_name_split[4] == 'rgb':
                    train_cameras += ['13']
        elif len(setting_name_split) == 2:
            if setting_name_split[1] == 'cctv':
                train_cameras += ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            elif setting_name_split[1] == 'uav':
                train_cameras += ['13', '14']
            elif setting_name_split[1] == 'ir':
                train_cameras += ['07', '08', '09', '10', '11', '12', '14']
            elif setting_name_split[1] == 'rgb':
                train_cameras += ['01', '02', '03', '04', '05', '06', '13']
        else:
            print("!!!setting name error!!!!")

        samples = []
        id_train.extend(id_val)

        for pid_str in sorted(id_train):
            for cam_str in train_cameras:
                img_dir = osp.join(dir_path, cam_str, pid_str)
                if osp.isdir(img_dir):
                    for fname in sorted(os.listdir(img_dir)):
                        img_path = osp.join(img_dir, fname)
                        if osp.isfile(img_path):
                            samples.append((img_path, int(pid_str), int(cam_str)))

        pid_container = set(pid for _, pid, _ in samples)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path, pid, camid in samples:
            if relabel:
                pid = pid2label[pid]
            viewid = camid -1
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
    
    def _process_query(self, dir_path, file_path, exp_setting=None, relabel=False):
        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        setting_name_split = exp_setting.split("_")
        if setting_name_split[1] == 'cctv' and setting_name_split[2] == 'ir':
            query_cameras = ['07', '08', '09', '10', '11', '12']
        elif setting_name_split[1] == 'cctv' and setting_name_split[2] == 'rgb':
            query_cameras = ['01', '02', '03', '04', '05', '06']
        elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'ir':
            query_cameras = ['14']
        elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'rgb':
            query_cameras = ['13']
        else:
            print("!!!setting name error!!!!")
            query_cameras = []

        samples = []
        for pid_str in sorted(ids):
            for cam_str in query_cameras:
                img_dir = osp.join(dir_path, cam_str, pid_str)
                if osp.isdir(img_dir):
                    for fname in sorted(os.listdir(img_dir)):
                        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            continue
                        img_path = osp.join(img_dir, fname)
                        if osp.isfile(img_path):
                            samples.append((img_path, int(pid_str), int(cam_str)))

        pid_container = set(pid for _, pid, _ in samples)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path, pid, camid in samples:
            if relabel:
                pid = pid2label[pid]
            viewid = camid -1
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset

    def _process_gallery(self, dir_path, file_path, exp_setting=None, relabel=False):
        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        setting_name_split = exp_setting.split("_")
        if setting_name_split[3] == 'cctv' and setting_name_split[4] == 'ir':
            gallery_cameras = ['07', '08', '09', '10', '11', '12']
        elif setting_name_split[3] == 'cctv' and setting_name_split[4] == 'rgb':
            gallery_cameras = ['01', '02', '03', '04', '05', '06']
        elif setting_name_split[3] == 'uav' and setting_name_split[4] == 'ir':
            gallery_cameras = ['14']
        elif setting_name_split[3] == 'uav' and setting_name_split[4] == 'rgb':
            gallery_cameras = ['13']
        else:
            print("!!!setting name error!!!!")
            gallery_cameras = []

        samples = []
        for pid_str in sorted(ids):
            for cam_str in gallery_cameras:
                img_dir = osp.join(dir_path, cam_str, pid_str)
                if osp.isdir(img_dir):
                    for fname in sorted(os.listdir(img_dir)):
                        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            continue
                        img_path = osp.join(img_dir, fname)
                        if osp.isfile(img_path):
                            samples.append((img_path, int(pid_str), int(cam_str)))

        pid_container = set(pid for _, pid, _ in samples)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path, pid, camid in samples:
            if relabel:
                pid = pid2label[pid]
            viewid = camid -1
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
    
    def _process_same(self, dir_path, file_path, exp_setting=None, relabel=False, split_ratio=0.5):
        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids] 

        setting_name_split = exp_setting.split("_")
        if setting_name_split[1] == 'cctv':
            cameras = ['01','02','03','04','05','06','07','08','09','10','11','12']
        elif setting_name_split[1] == 'uav':
            cameras = ['13','14']
        elif setting_name_split[1] == 'ir':
            cameras = ['07','08','09','10','11','12','14']
        elif setting_name_split[1] == 'rgb':
            cameras = ['01','02','03','04','05','06','13']
        else:
            print("!!!setting name error!!!!")
        
        all_files = []
        for id in sorted(ids):
            for cam in cameras:
                img_dir = osp.join(dir_path, cam, id)
                if osp.isdir(img_dir):
                    new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                    all_files.extend(new_files)
        
        pid_container = set()
        for img_path in all_files:
            pid = int(img_path[-13:-9])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        
        query_files = []
        gallery_files = []
        
        id_cam_files = defaultdict(list)
        for img_path in all_files:
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            id_cam_files[(pid, camid)].append(img_path)
        
        for (pid, camid), files in id_cam_files.items():
            if len(files) == 1:
                continue
            random_files = files.copy()
            random.shuffle(random_files)
            split_point = max(1, int(len(files) * split_ratio))
            
            for img_path in random_files[:split_point]:
                if relabel:
                    labeled_pid = pid2label[pid]
                else:
                    labeled_pid = pid
                viewid = camid - 1
                query_files.append((img_path, self.pid_begin + labeled_pid, camid, viewid))
                
            for img_path in random_files[split_point:]:
                if relabel:
                    labeled_pid = pid2label[pid]
                else:
                    labeled_pid = pid
                viewid = camid - 1
                gallery_files.append((img_path, self.pid_begin + labeled_pid, camid, viewid))

        # query_pids = set(pid for _, pid, _, _ in query_files)
        # gallery_pids = set(pid for _, pid, _, _ in gallery_files)
        # print('query_pids', len(query_pids))
        # print('gallery_pids', len(gallery_pids))
        # print('query_pids - gallery_pids', query_pids - gallery_pids)
        # print('gallery_pids - query_pids', gallery_pids - query_pids)
        # breakpoint()
        
        return query_files, gallery_files

if __name__ == '__main__':
    dataset = MMMP(exp_setting='exp_cctv_rgb_uav_rgb')
    print(dataset.num_train_pids)
    print(dataset.num_query_pids)
    print(dataset.num_gallery_pids)
    print(dataset.train[1])


    #代码中原来处理训练集的代码
    # def _process_train(self, dir_path, file_path_train, file_path_val, exp_setting=None, relabel=True):
    # with open(file_path_train, 'r') as file:
    #     ids = file.read().splitlines()
    #     ids = [int(y) for y in ids[0].split(',')]
    #     id_train = ["%04d" % x for x in ids]

    # with open(file_path_val, 'r') as file:
    #     ids = file.read().splitlines()
    #     ids = [int(y) for y in ids[0].split(',')]
    #     id_val = ["%04d" % x for x in ids]

    # setting_name_split = exp_setting.split("_")
    # train_cameras = []
    # if len(setting_name_split) == 5:
    #     if setting_name_split[1] == 'cctv':
    #         if setting_name_split[2] == 'ir':
    #             train_cameras += ['07', '08', '09', '10', '11', '12']
    #         elif setting_name_split[2] == 'rgb':
    #             train_cameras += ['01', '02', '03', '04', '05', '06']
    #     elif setting_name_split[1] == 'uav':
    #         if setting_name_split[2] == 'ir':
    #             train_cameras += ['14']
    #         elif setting_name_split[2] == 'rgb':
    #             train_cameras += ['13']

    #     if setting_name_split[3] == 'cctv':
    #         if setting_name_split[4] == 'ir':
    #             train_cameras += ['07', '08', '09', '10', '11', '12']
    #         elif setting_name_split[4] == 'rgb':
    #             train_cameras += ['01', '02', '03', '04', '05', '06']
    #     elif setting_name_split[3] == 'uav':
    #         if setting_name_split[4] == 'ir':
    #             train_cameras += ['14']
    #         elif setting_name_split[4] == 'rgb':
    #             train_cameras += ['13']
    # elif len(setting_name_split) == 2:
    #     if setting_name_split[1] == 'cctv':
    #         train_cameras += ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    #     elif setting_name_split[1] == 'uav':
    #         train_cameras += ['13', '14']
    #     elif setting_name_split[1] == 'ir':
    #         train_cameras += ['07', '08', '09', '10', '11', '12', '14']
    #     elif setting_name_split[1] == 'rgb':
    #         train_cameras += ['01', '02', '03', '04', '05', '06', '13']
    # else:
    #     print("!!!setting name error!!!!")

    # samples = []
    # id_train.extend(id_val)

    # for pid_str in sorted(id_train):
    #     for cam_str in train_cameras:
    #         img_dir = osp.join(dir_path, cam_str, pid_str)
    #         if osp.isdir(img_dir):
    #             for fname in sorted(os.listdir(img_dir)):
    #                 img_path = osp.join(img_dir, fname)
    #                 if osp.isfile(img_path):
    #                     samples.append((img_path, int(pid_str), int(cam_str)))

    # pid_container = set(pid for _, pid, _ in samples)
    # pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

    # dataset = []
    # for img_path, pid, camid in samples:
    #     if relabel:
    #         pid = pid2label[pid]
    #     dataset.append((img_path, self.pid_begin + pid, camid, 0))

    # return dataset

    #代码中原来处理查询集的代码
    # def _process_query(self, dir_path,file_path, exp_setting = None,relabel=False):
    
    # with open(file_path, 'r') as file:
    #     ids = file.read().splitlines()
    #     ids = [int(y) for y in ids[0].split(',')]
    #     ids = ["%04d" % x for x in ids] 


    # setting_name_split = exp_setting.split("_")
    # if setting_name_split[1] == 'cctv' and setting_name_split[2] == 'ir':
    #     query_cameras = ['07','08','09','10','11','12']
    # elif setting_name_split[1] == 'cctv' and setting_name_split[2] == 'rgb':
    #     query_cameras = ['01','02','03','04','05','06']
    # elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'ir':
    #     query_cameras = ['14']
    # elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'rgb':
    #     query_cameras = ['13']
    # else:
    #     print("!!!setting name error!!!!")
    # files = []
    # for id in sorted(ids):
    #     for cam in query_cameras:
    #         img_dir = osp.join(dir_path,cam,id)
    #         if osp.isdir(img_dir):
    #             new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
    #             files.extend(new_files)      
    # pid_container = set()
    # for img_path in files:
    #     pid = int(img_path[-13:-9])
    #     pid_container.add(pid)
    # pid2label = {pid:label for label, pid in enumerate(pid_container)}
    
    # dataset = []
    # for img_path in files:
    #     camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        
    #     if relabel: pid = pid2label[pid]

    #     dataset.append((img_path, self.pid_begin + pid, camid, 0))
    # return dataset


        #代码中原来处理图库集的代码
        # def _process_gallery(self, dir_path,file_path,exp_setting = None, relabel=False):
        
        # with open(file_path, 'r') as file:
        #     ids = file.read().splitlines()
        #     ids = [int(y) for y in ids[0].split(',')]
        #     ids = ["%04d" % x for x in ids] 


        # setting_name_split = exp_setting.split("_")
        # if setting_name_split[3] == 'cctv' and setting_name_split[4] == 'ir':
        #     gallery_cameras = ['07','08','09','10','11','12']
        # elif setting_name_split[3] == 'cctv' and setting_name_split[4] == 'rgb':
        #     gallery_cameras = ['01','02','03','04','05','06']
        # elif setting_name_split[3] == 'uav' and setting_name_split[4] == 'ir':
        #     gallery_cameras = ['14']
        # elif setting_name_split[3] == 'uav' and setting_name_split[4] == 'rgb':
        #     gallery_cameras = ['13']
        # else:
        #     print("!!!setting name error!!!!")
        # files = []
        # for id in sorted(ids):
        #     for cam in gallery_cameras:
        #         img_dir = osp.join(dir_path,cam,id)
        #         if osp.isdir(img_dir):
        #             new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
        #             files.extend(new_files)      
        # pid_container = set()
        # for img_path in files:
        #     pid = int(img_path[-13:-9])
        #     pid_container.add(pid)
        # pid2label = {pid:label for label, pid in enumerate(pid_container)}
       
        # dataset = []
        # for img_path in files:
        #     camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            
        #     if relabel: pid = pid2label[pid]

        #     dataset.append((img_path, self.pid_begin + pid, camid, 0))
        # return dataset
