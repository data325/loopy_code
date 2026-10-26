from functools import partial

import numpy as np
from PIL import Image

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def disable_augmentation(self, augmentor_configs):
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
             
    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def v2x_sensor_vfield(self, data_dict=None, config=None):
        """Object-local smooth vector fields plus roadside LiDAR randomization.

        The object deformation follows the 3D-VField principle: points within
        boxes receive a bounded, spatially smooth displacement while labels
        remain fixed. Sensor noise models beam/range variation seen across
        roadside scanners without using target labels in detector training.
        """
        if data_dict is None:
            return partial(self.v2x_sensor_vfield, config=config)
        points = data_dict['points'].copy()
        boxes = data_dict['gt_boxes']
        probability = float(config.get('PROBABILITY', 0.8))
        if np.random.rand() < probability:
            max_deform = float(config.get('MAX_DEFORM', 0.08))
            for box in boxes:
                rel = points[:, :3] - box[None, :3]
                c, s = np.cos(box[6]), np.sin(box[6])
                local = np.stack([c * rel[:, 0] + s * rel[:, 1],
                                  -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=1)
                half = np.maximum(box[3:6] / 2, 1e-3)
                norm = local / half[None]
                inside = (np.abs(norm) <= 1).all(axis=1)
                if not inside.any():
                    continue
                phase = np.random.uniform(-np.pi, np.pi, 3)
                amplitude = np.random.uniform(-max_deform, max_deform, 3)
                field = amplitude[None] * np.sin(np.pi * norm[inside] + phase[None])
                taper = np.prod(np.clip(1 - norm[inside] ** 2, 0, 1), axis=1, keepdims=True)
                local[inside] += field * taper
                points[inside, 0] = box[0] + c * local[inside, 0] - s * local[inside, 1]
                points[inside, 1] = box[1] + s * local[inside, 0] + c * local[inside, 1]
                points[inside, 2] = box[2] + local[inside, 2]
            xyz = points[:, :3]
            ranges = np.linalg.norm(xyz, axis=1)
            sigma = float(config.get('RANGE_NOISE_STD', 0.02))
            scale = 1 + np.random.normal(0, sigma, len(points)).astype(np.float32) / np.maximum(ranges, 1.0)
            points[:, :3] *= scale[:, None]
            drop_min, drop_max = config.get('DROP_RATE', [0.0, 0.12])
            drop = np.random.uniform(drop_min, drop_max)
            range_factor = np.clip(ranges / 80.0, 0, 1)
            keep = np.random.rand(len(points)) > drop * (0.35 + 0.65 * range_factor)
            points = points[keep]
        data_dict['points'] = points
        return data_dict




    def v2x_ped_align(self, data_dict=None, config=None):
        """Reshape synthetic pedestrians toward real V2X-Real box statistics.

        Real roadside pedestrians are ~0.74 x 0.72 x 1.73 m; the CARLA
        synthetic ones are ~1.84 x 0.80 x 1.61 m. The width mismatch alone
        caps IoU@0.5 near 0.35 even with perfect localization, so with
        probability PROBABILITY each pedestrian box (and the points inside
        it) is rescaled toward the real size distribution. Labels stay
        consistent with the transformed points (synthetic-only supervision).
        """
        if data_dict is None:
            return partial(self.v2x_ped_align, config=config)
        gt_names = data_dict.get('gt_names', None)
        if gt_names is None or len(gt_names) == 0:
            return data_dict
        points = data_dict['points'].copy()
        boxes = data_dict['gt_boxes']
        prob = float(config.get('PROBABILITY', 0.85))
        real_dims = np.asarray(config.get('REAL_DIMS', [0.74, 0.72, 1.73]),
                               dtype=np.float32)
        dim_std = np.asarray(config.get('DIM_STD', [0.09, 0.09, 0.12]),
                             dtype=np.float32)
        keep_orig = float(config.get('KEEP_ORIGINAL_PROB', 0.25))
        sparse_prob = float(config.get('SPARSE_PROB', 0.0))
        sparse_wedge_max = int(config.get('SPARSE_WEDGE_MAX', 2))
        sparse_wedge_half_w_max = float(config.get('SPARSE_WEDGE_HALF_W_MAX', 0.45))
        sparse_frac_lo = float(config.get('SPARSE_FRAC_LO', 0.55))
        slab_prob = float(config.get('SLAB_PROB', 0.0))
        side_prob = float(config.get('SIDE_PROB', 0.0))
        lo = np.asarray(config.get('CLIP_LO', [0.45, 0.45, 1.25]),
                        dtype=np.float32)
        hi = np.asarray(config.get('CLIP_HI', [1.05, 1.05, 2.10]),
                        dtype=np.float32)
        keep_mask = np.ones(len(points), dtype=bool)
        for i, name in enumerate(gt_names):
            if name != 'pedestrian':
                continue
            if np.random.rand() >= prob or np.random.rand() < keep_orig:
                continue
            box = boxes[i]
            old = np.maximum(box[3:6], 1e-3)
            new = np.clip(np.random.normal(real_dims, dim_std), lo, hi)
            scale = new / old
            rel = points[:, :3] - box[None, :3]
            c, s = np.cos(box[6]), np.sin(box[6])
            local = np.stack([c * rel[:, 0] + s * rel[:, 1],
                              -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]],
                             axis=1)
            half = old / 2
            inside = (np.abs(local) <= half[None]).all(axis=1)
            idx = np.nonzero(inside)[0]
            if len(idx) == 0:
                continue
            local[idx] *= scale[None]
            points[idx, 0] = box[0] + c * local[idx, 0] - s * local[idx, 1]
            points[idx, 1] = box[1] + s * local[idx, 0] + c * local[idx, 1]
            points[idx, 2] = box[2] + local[idx, 2]
            boxes[i, 3:6] = new
            if sparse_prob > 0.0 and np.random.rand() < sparse_prob:
                # simulate sparse / occluded roadside returns: drop 1-2
                # azimuth wedges plus a random point fraction
                az = np.arctan2(points[idx, 1], points[idx, 0])
                n_wedges = int(np.random.randint(1, sparse_wedge_max + 1))
                for _ in range(n_wedges):
                    center = np.random.uniform(-np.pi, np.pi)
                    half_w = np.random.uniform(0.15, sparse_wedge_half_w_max)
                    delta = np.abs((az - center + np.pi) % (2 * np.pi) - np.pi)
                    keep_mask[idx[delta < half_w]] = False
                frac = np.random.uniform(sparse_frac_lo, 1.0)
                drop_extra = np.random.rand(len(idx)) > frac
                keep_mask[idx[drop_extra]] = False
                if slab_prob > 0.0 and np.random.rand() < slab_prob:
                    # drop a horizontal slab (bottom or middle) of the
                    # pedestrian: simulates occlusion by vehicles/objects
                    z_local = points[idx, 2] - box[2]
                    h = max(box[5], 1e-3)
                    z_lo = np.random.uniform(-h/2, -h/8)
                    z_hi = z_lo + np.random.uniform(0.15, 0.45)
                    keep_mask[idx[(z_local >= z_lo) & (z_local <= z_hi)]] = False
                if side_prob > 0.0 and np.random.rand() < side_prob:
                    # one-side lateral occlusion: drop most points on one
                    # lateral side (local y), forcing the model to estimate
                    # the center from asymmetric/partial evidence
                    lat = -s * (points[idx, 0] - box[0]) + c * (points[idx, 1] - box[1])
                    side = np.random.choice([-1.0, 1.0])
                    side_mask = (lat * side) > 0
                    drop_frac = np.random.uniform(0.5, 0.85)
                    rnd = np.random.rand(len(idx)) < drop_frac
                    keep_mask[idx[side_mask & rnd]] = False
        points = points[keep_mask]
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, return_flip=True
            )
            data_dict['flip_%s'%cur_axis] = enable
            if 'roi_boxes' in data_dict.keys():
                num_frame, num_rois,dim = data_dict['roi_boxes'].shape
                roi_boxes, _, _ = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                data_dict['roi_boxes'].reshape(-1,dim), np.zeros([1,3]), return_flip=True, enable=enable
                )
                data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )
        if 'roi_boxes' in data_dict.keys():
            num_frame, num_rois,dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = augmentor_utils.global_rotation(
            data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]), rot_range=rot_range, return_rot=True, noise_rotation=noise_rot)
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        
        if 'roi_boxes' in data_dict.keys():
            gt_boxes, roi_boxes, points, noise_scale = augmentor_utils.global_scaling_with_roi_boxes(
                data_dict['gt_boxes'], data_dict['roi_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )
            data_dict['roi_boxes'] = roi_boxes
        else:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        assert len(noise_translate_std) == 3
        noise_translate = np.array([
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[2], 1),
        ], dtype=np.float32).T

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        points[:, :3] += noise_translate
        gt_boxes[:, :3] += noise_translate
                
        if 'roi_boxes' in data_dict.keys():
            data_dict['roi_boxes'][:, :3] += noise_translate
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_translate'] = noise_translate
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                                 config['SWAP_PROB'],
                                                                 config['SWAP_MAX_NUM'],
                                                                 pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def imgaug(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imgaug, config=config)
        imgs = data_dict["camera_imgs"]
        img_process_infos = data_dict['img_process_infos']
        new_imgs = []
        for img, img_process_info in zip(imgs, img_process_infos):
            flip = False
            if config.RAND_FLIP and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*config.ROT_LIM)
            # aug images
            if flip:
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            img = img.rotate(rotate)
            img_process_info[2] = flip
            img_process_info[3] = rotate
            new_imgs.append(img)

        data_dict["camera_imgs"] = new_imgs
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
