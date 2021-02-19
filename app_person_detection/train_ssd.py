

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


import os
import logging
import sys
import itertools
import torch



from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_224
from vision.ssd.shufflenet_ssd_lite import creat_shufflenet_ssd_lite
from vision.nn.multibox_loss import MultiboxLoss, FocalLoss
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.config import mobilenetv1_ssd_config
from torchsummary import summary

from vision.utils.argument import _argument
from torchscope import scope
from vision.ssd.tiny_mobilenet_v2_ssd import create_Mb_Tiny_RFB_fd
from vision.ssd.config import tiny_mobilenet_v2_config
from train.train import train, test, data_loader

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
Data = [["/media/ducanh/DATA/tienln/data/human/use/wider_person/annotations","/media/ducanh/DATA/tienln/data/human/use/wider_person/images"],
        ["/media/ducanh/DATA/tienln/data/human/use/crowd_human/annotations","/media/ducanh/DATA/tienln/data/human/use/crowd_human/images"],
        ["/media/ducanh/DATA/tienln/data/MSCOCO/2017/annotations/train","/media/ducanh/DATA/tienln/data/MSCOCO/2017/train2017"],
        ["/media/ducanh/DATA/tienln/data/VOC2012/person_json_annotations","/media/ducanh/DATA/tienln/data/VOC2012/JPEGImages"],
        ["/media/ducanh/DATA/tienln/data/human/use/crowd_human/origin_annotations/val1","/media/ducanh/DATA/tienln/data/human/use/crowd_human/images"]]
args = _argument()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

if __name__ == '__main__':
    timer = Timer()
    logging.info(args)
    
    if args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    if args.net == 'tiny_mobilenet':
        create_net = create_Mb_Tiny_RFB_fd
        config = tiny_mobilenet_v2_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_loader, num_classes = data_loader(Data, config)


    logging.info("Build network.")
    net = create_net(num_classes)
    scope(net, (3, 224, 224))
    net.save('/media/ducanh/DATA/tienln/ai_camera/ssd_lite_person_detection/test.pth')
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    # criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
    #                          center_variance=0.1, size_variance=0.2, device=DEVICE)
    criterion = FocalLoss()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        scheduler.step()
        training_loss = train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:

            # val_running_loss, val_running_regression_loss, val_running_classification_loss = test(val_loader,net,criterion,device=DEVICE)
            # logging.info(
            #     f"Epoch: {epoch}, " +
            #     f"val_avg_loss: {val_running_loss:.4f}, " +
            #     f"val_reg_loss {val_running_regression_loss:.4f}, " +
            #     f"val_cls_loss: {val_running_classification_loss:.4f}")

            # model_path = os.path.join(args.checkpoint_folder, f"{args.net}-epoch-{epoch}-train_loss-{round(training_loss,2)}-val_loss-{round(val_running_loss,2)}.pth")
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-epoch-{epoch}-train_loss-{round(training_loss,2)}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")