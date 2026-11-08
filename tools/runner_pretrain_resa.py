import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from datasets.dual_view_transforms import DualViewTransform, AdaptiveDualViewTransform

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils

train_transforms = transforms.Compose([
    data_transforms.PointcloudScaleAndTranslate(),
])


def safe_tensor_to_scalar(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    if hasattr(tensor, 'item'):
        if tensor.numel() == 1:
            return tensor.item()
        else:
            return tensor.mean().item()
    else:
        return float(tensor)


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    use_resa = getattr(config.model, 'use_resa', False)
    use_dual_view = getattr(config, 'use_dual_view', False)
    
    dual_view_config = getattr(config, 'dual_view_config', {}) if (use_resa and use_dual_view) else None
    if dual_view_config is not None:
        print_log(f'[ReSA] Using adaptive dual view augmentation (base config: {dual_view_config})', logger=logger)
    else:
        print_log('[ReSA] Using single view training (MAE only)', logger=logger)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model, 
            device_ids=[args.local_rank % torch.cuda.device_count()], 
            find_unused_parameters=True
        )
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        if use_resa:
            losses = AverageMeter(['Total', 'MAE', 'ReSA'])
        else:
            losses = AverageMeter(['Loss'])

        num_iter = 0
        n_batches = len(train_dataloader)
        
        if dual_view_config is not None:
            base_jitter = dual_view_config.get('jitter_std', 0.01)
            scale_low = dual_view_config.get('scale_low', 0.8)
            scale_high = dual_view_config.get('scale_high', 1.2)
            base_scale_range = max(1.0 - float(scale_low), float(scale_high) - 1.0)
            base_translate = dual_view_config.get('translate_range', 0.1)
            jitter_clip = dual_view_config.get('jitter_clip', None)
            dropout_ratio = dual_view_config.get('dropout_ratio', None)

            dual_view_transform = AdaptiveDualViewTransform(
                epoch=epoch,
                max_epochs=config.max_epoch,
                base_jitter=base_jitter,
                base_scale_range=base_scale_range,
                base_translate=base_translate,
                jitter_clip=jitter_clip,
                dropout_ratio=dropout_ratio,
            )
        else:
            dual_view_transform = None

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            
            if dataset_name == 'ShapeNet':
                points = data.cuda()
                points = misc.fps(points, npoints)
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            
            if use_resa and dual_view_transform is not None:
                result = base_model(points, epoch=epoch, view_transform=dual_view_transform)
                
                if isinstance(result, dict):
                    total_loss = result['total_loss']
                    mae_loss = result['mae_loss'] 
                    resa_loss = result['resa_loss']
                    resa_weight = result.get('resa_weight', 0.0)
                    
                    if args.distributed:
                        total_loss_reduced = dist_utils.reduce_tensor(total_loss, args)
                        mae_loss_reduced = dist_utils.reduce_tensor(mae_loss, args)
                        resa_loss_reduced = dist_utils.reduce_tensor(resa_loss, args)
                        losses.update([
                            safe_tensor_to_scalar(total_loss_reduced) * 1000,
                            safe_tensor_to_scalar(mae_loss_reduced) * 1000, 
                            safe_tensor_to_scalar(resa_loss_reduced) * 1000
                        ])
                    else:
                        losses.update([
                            safe_tensor_to_scalar(total_loss) * 1000,
                            safe_tensor_to_scalar(mae_loss) * 1000,
                            safe_tensor_to_scalar(resa_loss) * 1000
                        ])
                    
                    loss = total_loss
                else:
                    loss = result
                    if args.distributed:
                        loss_reduced = dist_utils.reduce_tensor(loss, args)
                        losses.update([
                            safe_tensor_to_scalar(loss_reduced) * 1000, 
                            safe_tensor_to_scalar(loss_reduced) * 1000,
                            0.0
                        ])
                    else:
                        losses.update([
                            safe_tensor_to_scalar(loss) * 1000,
                            safe_tensor_to_scalar(loss) * 1000,
                            0.0
                        ])
                    resa_weight = 0.0
            else:
                points = train_transforms(points)
                loss = base_model(points)
                
                if args.distributed:
                    loss = dist_utils.reduce_tensor(loss, args)
                    losses.update([safe_tensor_to_scalar(loss) * 1000])
                else:
                    losses.update([safe_tensor_to_scalar(loss) * 1000])

            try:
                loss.backward()
            except:
                loss = loss.mean()
                loss.backward()

            grad_clip = getattr(config, 'grad_clip', None)
            check_nan = getattr(config, 'check_nan', False)
            
            if check_nan:
                has_nan = False
                for name, param in base_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print_log(f'NaN/Inf detected in gradients of {name}', logger=logger)
                            has_nan = True
                            break
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            print_log(f'NaN/Inf detected in parameters of {name}', logger=logger)
                            has_nan = True
                            break
                
                if has_nan:
                    print_log('Skipping optimizer step due to NaN/Inf', logger=logger)
                    base_model.zero_grad()
                    continue
            
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip, norm_type=2)

            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                if use_resa:
                    train_writer.add_scalar('Loss/Batch/Total', losses.val()[0] / 1000, n_itr)
                    train_writer.add_scalar('Loss/Batch/MAE', losses.val()[1] / 1000, n_itr)
                    train_writer.add_scalar('Loss/Batch/ReSA', losses.val()[2] / 1000, n_itr)
                    if 'resa_weight' in locals():
                        weight_value = safe_tensor_to_scalar(resa_weight)
                        train_writer.add_scalar('Loss/Batch/ReSAWeight', weight_value, n_itr)
                else:
                    train_writer.add_scalar('Loss/Batch/Loss', losses.val()[0] / 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                if use_resa:
                    loss_str = f'Total={losses.val()[0]:.4f} MAE={losses.val()[1]:.4f} ReSA={losses.val()[2]:.4f}'
                    if 'resa_weight' in locals():
                        weight_value = safe_tensor_to_scalar(resa_weight)
                        loss_str += f' ReSAWeight={weight_value:.4f}'
                else:
                    loss_str = f'Loss={losses.val()[0]:.4f}'
                
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           loss_str, optimizer.param_groups[0]['lr']), logger=logger)
        
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        
        epoch_end_time = time.time()

        if train_writer is not None:
            if use_resa:
                train_writer.add_scalar('Loss/Epoch/Total', losses.avg(0), epoch)
                train_writer.add_scalar('Loss/Epoch/MAE', losses.avg(1), epoch)
                train_writer.add_scalar('Loss/Epoch/ReSA', losses.avg(2), epoch)
            else:
                train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        if use_resa:
            loss_str = f'Total={losses.avg(0):.4f} MAE={losses.avg(1):.4f} ReSA={losses.avg(2):.4f}'
        else:
            loss_str = f'Loss={losses.avg(0):.4f}'
        
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, loss_str,
                   optimizer.param_groups[0]['lr']), logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        
        if epoch % 25 == 0 and epoch >= 250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 
                                  f'ckpt-epoch-{epoch:03d}', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate_resa(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()
    
    test_losses = AverageMeter(['Loss'])
    test_metrics = AverageMeter(['ACC'])
    category_metrics = dict()
    npoints = config.dataset.val.others.npoints
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
                points = misc.fps(points, npoints)
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Validation phase do not support {dataset_name}')

            assert points.size(1) == npoints
            
            if hasattr(base_model.module if hasattr(base_model, 'module') else base_model, 'get_encoder_features'):
                features, centers = base_model.module.get_encoder_features(points, no_grad=True) if hasattr(base_model, 'module') else base_model.get_encoder_features(points, no_grad=True)
                
                
            loss = base_model(points)
            if isinstance(loss, dict):
                loss = loss['total_loss'] if 'total_loss' in loss else loss['mae_loss']
            
            test_losses.update([safe_tensor_to_scalar(loss)])

    print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, 0.0), logger=logger)
    
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', 0.0, epoch)
    
    return Acc_Metric(0.0)


def test_net():
    pass 
