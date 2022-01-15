import argparse


class Config(object):
    """
    Manually setting configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=300, type=int, help='number of epoch to train, same for each process.')
    parser.add_argument("--type", default='focal')
    parser.add_argument("--summary", default=True, type=bool, help='check if we use the summary')
    parser.add_argument("--backbone", default='resnet50', type=str, help='current only support for resnet50')
    parser.add_argument("--test_img", default=15, type=int)
    parser.add_argument("--lr", default=0.01, type=int, help='learning rate for model')
    args = parser.parse_args()
    backbone = args.backbone
    # loss_backend = args.loss_backend
    summary = args.summary
    epoch = args.epoch
    lr = args.lr
    type = args.type
    test_img = args.test_img
    # extend_program = args.extend
