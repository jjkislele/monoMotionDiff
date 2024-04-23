import os
import logging
import time
import numpy as np
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from einops import repeat

from models.model import DiffModel, adj_mx_from_edges
from models.backbone.GraFormer import GraFormer
from models.diffusion.ema import EMAHelper
from models.diffusion.utils import get_beta_schedule, generalized_steps
from utils.learning import get_optimizer, AverageMeter, lr_decay, define_error_list, test_calculation, print_error
from utils.loss import mpjpe, p_mpjpe
from dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
from dataset import read_3d_data, create_2d_data, fetch_kpts, PoseDataLoader


class DiffRunner(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = config.device
        self.model_diff = None
        self.model_pose = None
        self.subjects_train = TRAIN_SUBJECTS
        self.subjects_test = TEST_SUBJECTS
        self.mocap_dataset = None
        self.keypoints_train = None
        self.keypoints_test = None
        self.action_filter = None
        self.train_loader = None
        self.valid_loader = None
        self.flip_aug = False
        self.test_action_list = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing',
                                 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'Walking',
                                 'WalkTogether']

        # GraFormer mask
        self.src_mask = torch.tensor([[[True for _ in range(config.data.num_joints)]]]).cuda()

        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        # Augmentation
        self.kpt_left = config.data.kpt_left
        self.kpt_right = config.data.kpt_right
        if self.config.data.flip_aug == 1:
            self.flip_aug = True

    def create_diffusion_model(self, model_path=None):
        # Cross-channel 2D-3D feature adjacency
        edges_2d3d = torch.tensor([[0, 1], [1, 2], [2, 3],
                                   [0, 4], [4, 5], [5, 6],
                                   [0, 7], [7, 8], [8, 9], [9, 10],
                                   [8, 11], [11, 12], [12, 13],
                                   [8, 14], [14, 15], [15, 16]] + \
                                  [[i, i + 17] for i in range(17)] + \
                                  (np.array([[0, 1], [1, 2], [2, 3],
                                             [0, 4], [4, 5], [5, 6],
                                             [0, 7], [7, 8], [8, 9], [9, 10],
                                             [8, 11], [11, 12], [12, 13],
                                             [8, 14], [14, 15], [15, 16]]) + 17).tolist(),
                                  dtype=torch.long)
        adj_2d3d = adj_mx_from_edges(num_pts=34, edges=edges_2d3d, sparse=False)
        # 2D context adjacency
        edges_ctx = torch.tensor([[0, 1], [1, 2], [2, 3],
                                  [0, 4], [4, 5], [5, 6],
                                  [0, 7], [7, 8], [8, 9], [9, 10],
                                  [8, 11], [11, 12], [12, 13],
                                  [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj_ctx = adj_mx_from_edges(num_pts=17, edges=edges_ctx, sparse=False)
        self.model_diff = DiffModel(adj_2d3d=adj_2d3d.to(self.device), adj_ctx=adj_ctx.to(self.device),
                                    hid_dim=self.config.model.hid_dim,
                                    num_layers=self.config.model.num_layers,
                                    n_head=self.config.model.n_head,
                                    dropout=self.config.model.dropout,
                                    n_pts=self.config.model.n_pts * 2,
                                    coords_dim=(3, 3)).to(self.device)
        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])

    def create_pose_model(self):
        model_path = self.config.data.pose_model
        assert os.path.exists(model_path), f"Cannot find the lifter: {model_path}."
        # 2D-to-3D lifting model
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                              [0, 4], [4, 5], [5, 6],
                              [0, 7], [7, 8], [8, 9], [9, 10],
                              [8, 11], [11, 12], [12, 13],
                              [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        self.model_pose = GraFormer(adj=adj.to(self.device),
                                    hid_dim=self.config.model.hid_dim,
                                    num_layers=self.config.model.num_layers,
                                    n_head=self.config.model.n_head,
                                    dropout=self.config.model.dropout,
                                    n_pts=self.config.model.n_pts).cuda()
        self.model_pose = torch.nn.DataParallel(self.model_pose, device_ids=[0])

        # load pretrained model
        logging.info('initialize model by:' + model_path)
        states = torch.load(model_path)
        self.model_pose.load_state_dict(states[0], strict=False)

    def prepare_data(self, eval_only=False):
        args, config = self.args, self.config
        kpt_left = config.data.kpt_left
        kpt_right = config.data.kpt_right

        print(f'==> Using settings {args}')
        print(f'==> Using configures {config}')

        # load 3d gt
        dataset = Human36mDataset(config.data.dataset_path_3d)
        self.mocap_dataset = read_3d_data(dataset)
        # load 2d data
        self.keypoints_train = create_2d_data(config.data.dataset_path_2d, dataset)
        self.keypoints_test = create_2d_data(config.data.dataset_path_2d, dataset)
        # filter
        self.action_filter = None if args.actions == '*' else args.actions.split(',')
        if self.action_filter is not None:
            self.action_filter = map(lambda x: dataset.define_actions(x)[0], self.action_filter)
            print(f'==> Selected actions: {self.action_filter}')
        logging.info(f'-> Flip Aug: {self.flip_aug}')
        logging.info(f'-> Input 2D pose is {config.data.dataset_path_2d}')
        # create data loader
        # train
        if not eval_only:
            poses_train_3d, poses_train_2d, actions_train, act_cls_train = fetch_kpts(self.subjects_train,
                                                                                      self.mocap_dataset,
                                                                                      self.keypoints_train,
                                                                                      self.action_filter)
            self.train_loader = data.DataLoader(PoseDataLoader(poses_train_3d, poses_train_2d, actions_train,
                                                               act_cls_train, kpt_left=kpt_left, kpt_right=kpt_right,
                                                               aug=self.flip_aug),
                                                batch_size=config.training.batch_size, shuffle=True,
                                                num_workers=config.training.num_workers, pin_memory=True)
        # eval
        poses_valid_3d, poses_valid_2d, actions_valid, act_cls_valid = fetch_kpts(self.subjects_test,
                                                                                  self.mocap_dataset,
                                                                                  self.keypoints_test,
                                                                                  self.action_filter)
        self.valid_loader = data.DataLoader(PoseDataLoader(poses_valid_3d, poses_valid_2d, actions_valid,
                                                           act_cls_valid, kpt_left=kpt_left, kpt_right=kpt_right),
                                            batch_size=config.training.batch_size, shuffle=False,
                                            num_workers=config.training.num_workers, pin_memory=True)

    def train(self):
        cudnn.benchmark = True

        # init params
        best_p1, best_epoch = 1000, 0
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        start_epoch, step = 0, 0
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma

        # init diffusion
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None

        noise_scale = torch.from_numpy(np.array([0.1, 0.1, 1., 1., 1.])).cuda()

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            epoch_loss_diff = AverageMeter()

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()

            for i, (targets_3d, targets_2d, *_) in enumerate(self.train_loader):
                data_time += time.time() - data_start
                step += 1

                # to cuda
                targets_3d, targets_2d = targets_3d.to(self.device), targets_2d.to(self.device)
                targets_3d = targets_3d.unsqueeze(1)
                targets_2d = targets_2d.unsqueeze(1)

                # generate noisy sample based on selected time t and beta
                n, f, j, d = targets_3d.shape
                targets_uvxyz = torch.cat((targets_2d, targets_3d), dim=-1)
                targets_noise_scale = repeat(noise_scale, 'd -> b f j d', b=n, f=f, j=j)
                x = targets_uvxyz.clone()

                ##########################################
                e = torch.randn_like(x) * targets_noise_scale
                ##########################################

                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                # generate x_t (refer to DDIM equation)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()

                # predict noise
                output_noise = self.model_diff(x, targets_2d, self.src_mask, t.float())
                loss_diff = (e - output_noise).square().sum(dim=(1, 2, 3)).mean(dim=0)

                optimizer.zero_grad()
                loss_diff.backward()
                torch.nn.utils.clip_grad_norm_(self.model_diff.parameters(), self.config.optim.grad_clip)
                optimizer.step()
                epoch_loss_diff.update(loss_diff.item(), n)

                if self.config.model.ema:
                    ema_helper.update(self.model_diff)

                if i % 100 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'
                                 .format(epoch, i + 1, len(self.train_loader), step, data_time, epoch_loss_diff.avg))

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma)

            if epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states, os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                logging.info('test the performance of current model')

                p1, p2 = self.test(is_train=True)

                if p1 < best_p1:
                    best_p1 = p1
                    best_epoch = epoch
                logging.info('\n| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'
                             .format(best_epoch, best_p1, epoch, p1, p2))

    def test(self, is_train=False, is_store=False):
        cudnn.benchmark = True
        if is_store:
            pred_data = {}
        test_times = self.config.testing.test_times
        test_timesteps = self.config.testing.test_timesteps
        test_num_diffusion_timesteps = self.config.testing.test_num_diffusion_timesteps

        data_start = time.time()
        data_time = 0
        my_total_time_diff = 0
        my_total_time_pose = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        # get params
        model_params_diff = 0
        model_params_pose = 0
        for parameter in self.model_diff.parameters():
            model_params_diff += parameter.numel()
        logging.info('INFO: [Diff] parameter count: {}'.format(model_params_diff / 1000000, 'Million'))
        for parameter in self.model_pose.parameters():
            model_params_pose += parameter.numel()
        logging.info('INFO: [Pose] parameter count: {}'.format(model_params_pose / 1000000, 'Million'))
        if self.config.diffusion.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.config.diffusion.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        action_error_sum = define_error_list(self.test_action_list)

        for i, (targets_3d, targets_2d, input_action, input_action_with_idx) in enumerate(self.valid_loader):
            torch.cuda.synchronize()
            data_time += time.time() - data_start

            targets_3d, targets_2d = targets_3d.to(self.device), targets_2d.to(self.device)

            # build uvxyz
            my_time_start_pose = time.time()
            inputs_xyz = self.model_pose(targets_2d, self.src_mask)
            my_time_spent_pose = time.time() - my_time_start_pose
            my_total_time_pose += my_time_spent_pose
            inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :]

            # prepare the diffusion parameters
            # frame_num = 1
            inputs_xyz = inputs_xyz.unsqueeze(1)
            targets_2d = targets_2d.unsqueeze(1)

            ##########################################
            input_uvxyz = torch.cat((targets_2d, inputs_xyz), dim=-1)
            e = input_uvxyz.clone()
            e = e.repeat(test_times, 1, 1, 1)  # multiply batch size
            targets_2d = targets_2d.repeat(test_times, 1, 1, 1)
            ##########################################

            my_time_start_diff = time.time()
            output_3d = generalized_steps(e, targets_2d, self.src_mask,
                                          seq, self.model_diff, self.betas, eta=self.config.diffusion.eta)
            output_3d = output_3d[0][-1]
            output_3d = torch.mean(output_3d.reshape(test_times, -1, 1, 17, 5), 0)
            output_3d = output_3d[:, :, :, 2:]

            # normalize
            output_3d -= output_3d[:, :, :1]
            output_3d = output_3d.squeeze(1)

            if self.flip_aug:
                # flip, apply test-time-augmentation (following VideoPose3d)
                targets_2d_flip = targets_2d.clone()
                inputs_xyz_flip = inputs_xyz.clone()
                targets_2d_flip[:, :, :, 0] *= -1
                targets_2d_flip[:, :, self.kpt_left + self.kpt_right] = targets_2d_flip[:, :,
                                                                        self.kpt_right + self.kpt_left]
                inputs_xyz_flip[:, :, :, 0] *= -1
                inputs_xyz_flip[:, :, self.kpt_left + self.kpt_right] = inputs_xyz_flip[:, :,
                                                                        self.kpt_right + self.kpt_left]
                # prepare input data
                input_uvxyz_flip = torch.cat((targets_2d_flip, inputs_xyz_flip), dim=-1)
                e_flip = input_uvxyz_flip.clone()
                e_flip = e_flip.repeat(test_times, 1, 1, 1)  # multiply batch size
                targets_2d_flip = targets_2d_flip.repeat(test_times, 1, 1, 1)
                # diffuse
                output_3d_flip = generalized_steps(e_flip, targets_2d_flip, self.src_mask,
                                                   seq, self.model_diff, self.betas, eta=self.config.diffusion.eta)
                output_3d_flip = output_3d_flip[0][-1]
                output_3d_flip = torch.mean(output_3d_flip.reshape(test_times, -1, 1, 17, 5), 0)
                output_3d_flip = output_3d_flip[:, :, :, 2:]
                output_3d_flip -= output_3d_flip[:, :, :1]
                # un-flip
                output_3d_flip[:, :, :, 0] *= -1
                output_3d_flip[:, :, self.kpt_left + self.kpt_right] = output_3d_flip[:, :,
                                                                       self.kpt_right + self.kpt_left]
                # fuse
                output_3d_flip = output_3d_flip.squeeze(1)
                output_3d = (output_3d + output_3d_flip) / 2

            epoch_loss_3d_pos.update(mpjpe(output_3d, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_3d.cpu().numpy(),
                                                        targets_3d.cpu().numpy()).item() * 1000.0,
                                                targets_3d.size(0))
            torch.cuda.synchronize()
            data_start = time.time()
            my_time_spent_diff = time.time() - my_time_start_diff
            my_total_time_diff += my_time_spent_diff

            # jjk note: trim action label
            action_error_sum = test_calculation(output_3d, targets_3d, input_action, action_error_sum)

            # store prediction
            if is_store:
                pred_actions = output_3d.detach().cpu().numpy()
                for act_name, act in zip(input_action_with_idx, pred_actions):
                    if act_name not in pred_data:
                        pred_data[act_name] = []
                    pred_data[act_name].append(act)

            if i % 100 == 0 and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'
                             .format(batch=i + 1, size=len(self.valid_loader), data=data_time, e1=epoch_loss_3d_pos.avg,
                                     e2=epoch_loss_3d_pos_procrustes.avg))
                print('1: {}'.format(data_time))
                print('2: pose: {}. cur fps: {}'.format(my_total_time_pose, i / my_total_time_pose))
                print('2: diff: {}. cur fps: {}'.format(my_total_time_diff, i / my_total_time_diff))
        logging.info('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'
                     .format(batch=i + 1, size=len(self.valid_loader), data=data_time, e1=epoch_loss_3d_pos.avg,
                             e2=epoch_loss_3d_pos_procrustes.avg))

        p1, p2 = print_error(action_error_sum, is_train)

        if is_store:
            np.save(os.path.join(self.args.log_path, 'pred.npy'), pred_data)
        print(my_total_time_diff)

        return p1, p2
