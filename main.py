# Copyright (c) QIU Tian. All rights reserved.

import argparse
import datetime
import json
import os
import sys
import time

import torch
from termcolor import cprint

from engine import evaluate, train_one_epoch
from qtcls import __info__, dataloaders, build_criterion, build_dataset, build_model, build_optimizer, build_scheduler
from qtcls.utils.io import checkpoint_saver, checkpoint_loader, variables_loader, variables_saver, log_writer
from qtcls.utils.misc import init_seeds, get_n_params, mask_to_index
from qtcls.utils.os import makedirs, rmtree


def get_args_parser():
    parser = argparse.ArgumentParser('QTClassification', add_help=False)

    parser.add_argument('--config', '-c', type=str)

    # runtime
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--clip_max_norm', default=1.0, type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', action='store_true', help='evaluate only')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--drop_lr_now', action='store_true')
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')

    # dataset
    parser.add_argument('--dataset', '-d', type=str, default='amazon_homo', help='dataset name')

    # model
    parser.add_argument('--model', '-m', default='pagnn', type=str, help='model name')
    parser.add_argument('--model_kwargs', default=dict(), help='model specific kwargs')

    # criterion
    parser.add_argument('--criterion', default='default', type=str, help='criterion name')

    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer name')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', default=5e-2, type=float)

    # scheduler
    parser.add_argument('--scheduler', default='cosine', type=str, help='scheduler name')
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--warmup_lr', default=1e-6, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float, help='for CosineLR')

    # evaluator
    parser.add_argument('--evaluator', default='default', type=str, help='evaluator name')

    # loading
    parser.add_argument('--pretrain', '-p', type=str, help='path to the pre-trained weights (highest priority)')
    parser.add_argument('--no_pretrain', action='store_true', help='forcibly not use the pre-trained weights')
    parser.add_argument('--resume', '-r', type=str, help='checkpoint path to resume from')

    # saving
    parser.add_argument('--output_dir', '-o', type=str, default='./runs/__tmp__', help='path to save checkpoints, etc')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--clear_output_dir', '-co', action='store_true', help='clear output dir first')

    # remarks
    parser.add_argument('--note', type=str)

    return parser


def main(args):
    init_seeds(args.seed)

    cprint(__info__, 'light_green', attrs=['bold'])

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    meta_note = f'dataset: {args.dataset} | model: {args.model} | output_dir: {args.output_dir}'

    if device.type == 'cpu' or args.eval:
        args.amp = False
    if args.num_workers is None:
        args.num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    if args.resume:
        args.no_pretrain = True
    if args.no_pretrain:
        args.pretrain = None
    if args.clear_output_dir:
        rmtree(args.output_dir, not_exist_ok=True)
    if args.output_dir:
        makedirs(args.output_dir, exist_ok=True)
        variables_saver(dict(sorted(vars(args).items())), os.path.join(args.output_dir, 'config.py'))

    print(args)

    # ** dataset **
    graph = build_dataset(args)
    graph = graph.to(device)

    data_loader_train = dataloaders.NodeLoader(node_list=mask_to_index(graph.ndata['train_mask']),
                                               labels=graph.ndata['label'][graph.ndata['train_mask']],
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=False)

    data_loader_val = dataloaders.NodeLoader(node_list=mask_to_index(graph.ndata['val_mask']),
                                             labels=graph.ndata['label'][graph.ndata['val_mask']],
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False)

    data_loader_test = dataloaders.NodeLoader(node_list=mask_to_index(graph.ndata['test_mask']),
                                              labels=graph.ndata['label'][graph.ndata['test_mask']],
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=False)

    # ** model **
    model = build_model(args)
    model.to(device)

    n_params = get_n_params(model)
    print(f'#Params: {n_params / 1e6} M')

    # ** optimizer **
    param_dicts = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = build_optimizer(args, param_dicts)

    # ** criterion **
    criterion = build_criterion(args)

    # ** scheduler **
    scheduler = build_scheduler(args, optimizer, len(data_loader_train))

    # ** scaler **
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint_loader(model, checkpoint['model'], delete_keys=())
        if not args.eval and 'optimizer' in checkpoint and 'scheduler' in checkpoint and 'epoch' in checkpoint:
            checkpoint_loader(optimizer, checkpoint['optimizer'])
            checkpoint_loader(scheduler, checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.drop_lr_now:  # only works when using StepLR or MultiStepLR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
        if scaler and 'scaler' in checkpoint:
            checkpoint_loader(scaler, checkpoint["scaler"])

    if args.eval:
        print('\n')
        test_stats, evaluator = evaluate(
            graph, model, data_loader_test, criterion, device, args, args.print_freq, args.amp
        )
        return

    print('\n' + 'Start training:')
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            graph, model, criterion, data_loader_train, optimizer, scheduler, device, epoch, args.clip_max_norm, scaler,
            args.print_freq
        )
        if args.output_dir and (epoch + 1) % args.save_interval == 0:
            checkpoint_paths = [
                os.path.join(args.output_dir, f'checkpoint.pth'),
                os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth')
            ]
            for checkpoint_path in checkpoint_paths:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if scaler:
                    checkpoint['scaler'] = scaler.state_dict()
                checkpoint_saver(checkpoint, checkpoint_path)

        test_stats, evaluator = evaluate(
            graph, model, data_loader_test, criterion, device, args, args.print_freq, args.amp
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_params': n_params}

        if args.output_dir:
            log_writer(os.path.join(args.output_dir, 'log.txt'), json.dumps(log_stats))

        if args.note:
            print(f'{meta_note} | note: {args.note}\n')
        else:
            print(f'{meta_note}\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('QTClassification', parents=[get_args_parser()])
    argv = sys.argv[1:]

    idx = argv.index('-c') if '-c' in argv else (argv.index('--config') if '--config' in argv else -1)
    if idx not in [-1, len(argv) - 1] and not argv[idx + 1].startswith('-'):
        idx += 1

    args = parser.parse_args(argv[:idx + 1])

    if args.config:
        cfg = variables_loader(args.config)
        for k, v in cfg.items():
            setattr(args, k, v)

    args = parser.parse_args(argv[idx + 1:], args)

    main(args)
