import argparse
import json
import os
import os.path
import pkgutil


def get_format(dataset_name):
    json_data = pkgutil.get_data(__name__, "static/dataset_format.json")
    dataset_format = json.loads(json_data)

    return dataset_format[str(dataset_name)]


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.parse_args(open(values).read().split(), namespace)


def str_list(arg):
    return arg.split(',')


def int_list(arg):
    if arg:
        return list(map(int, arg.split(',')))
    else:
        return []


def float_list(arg):
    return list(map(float, arg.split(',')))


def lim_list(arg):
    lim = list(map(float, arg.split(',')))
    assert len(lim) == 2
    return lim


def interv_list(arg):
    interv_list = []
    for interv in arg.split(','):
        if "-" in interv:
            intervals = interv.split("-")
            interv_list += range(int(intervals[0]), int(intervals[1]) + 1)
        else:
            interv_list.append(int(interv))
    return interv_list


def global_args(parser, arg_file=None, prog_func=None):
    import torch

    if arg_file is None:
        import sys
        argv = sys.argv[1:]
    else:
        argv = ["--load-from-file", arg_file]

    global progress_fwd
    progress_fwd = prog_func

    args = parser.parse_args(argv)

    args_dict = vars(args)
    for arg in args_dict:
        globals()[arg] = args_dict[arg]

    torch.backends.cudnn.benchmark = True
    globals()[device] = torch.device(device)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def set_common_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/',
                            help="Root directory containing the climate datasets")
    arg_parser.add_argument('--log-dir', type=str, default='logs/', help="Directory where the log files will be stored")
    arg_parser.add_argument('--img-names', type=str_list, default='train.nc',
                            help="Comma separated list of netCDF files (climate dataset)")
    arg_parser.add_argument('--img-sizes', type=int_list, default='72,72',
                            help="Tuple of the size of the data (height x width)")
    arg_parser.add_argument('--data-types', type=str_list, default='tas', help="Comma separated list of variable types")
    arg_parser.add_argument('--device', type=str, default='cuda', help="Device used by PyTorch (cuda or cpu)")
    arg_parser.add_argument('--weights', type=str, default=None, help="Initialization weight")
    arg_parser.add_argument('--random-seed', type=int, default=None,
                            help="Random seed for iteration loop and initialization weights")
    arg_parser.add_argument('--progress-fwd', action='store_true', help="Print the progress of the forward propagation")
    arg_parser.add_argument('--normalization', type=str, default=None,
                            help="None: No normalization, "
                                 "std: normalize to 0 mean and 1 std, "
                                 "img: normalize values between -1 and 1 and 0.5 mean and 0.5 std, "
                                 "custom: normalize with custom define mean and std values")
    arg_parser.add_argument('--custom-mean', type=float, default=2e-4, help="Custom mean for image normalization")
    arg_parser.add_argument('--custom-std', type=float, default=2e-4, help="Custom std for image normalization")
    arg_parser.add_argument('--seed-size', type=int, default=100, help="Name of the dataset for format checking")
    arg_parser.add_argument('--n-layers', type=int, default=6, help="Number of layers")
    arg_parser.add_argument('--n-dense-layers', type=int, default=1, help="Number of dense layers in discriminator")
    arg_parser.add_argument('--gen-conv-factor', type=int, default=64,
                            help="Factor of convolutions for increasing model complexity")
    arg_parser.add_argument('--dis-conv-factor', type=int, default=64,
                            help="Factor of convolutions for increasing model complexity")
    arg_parser.add_argument('--attention-res', type=int_list, default=[],
                            help="Resolutions of current tensors where attention layers should apply")
    arg_parser.add_argument('--gan-type', type=str, default="default",
                            help="default: use simple GAN architecture, "
                                 "sa-gan: use Self-Attention GAN architecture, "
                                 "big-gan: use Big-GAN architecture")
    arg_parser.add_argument('--freva-project', type=str, default=None, help="Read data via freva project")
    arg_parser.add_argument('--freva-model', type=str, default=None, help="Read data via freva model")
    arg_parser.add_argument('--freva-experiment', type=str, default=None, help="Read data via freva experiment")
    arg_parser.add_argument('--freva-time-frequency', type=str, default=None, help="Read data via freva time-frequency")

    arg_parser.add_argument('--n-classes', type=int_list, default=None,
                            help="Comma separated list of integers defining the number of classes of different labels")
    arg_parser.add_argument('--class-dim', type=int, default=16, help="Number of channels for each conditioned label")

    arg_parser.add_argument('--locations', type=str_list, default=None,
                            help="Comma separated list of classes for location prediction")
    arg_parser.add_argument('--gt-ensembles', type=str_list, default=None,
                            help="Comma separated list of classes for location prediction")
    arg_parser.add_argument('--train-ssis', type=float_list, default=None,
                            help="Comma separated list of ssi values that are used for training")
    arg_parser.add_argument('--train-ensembles', type=interv_list, default=None,
                            help="Comma separated list of ensembles that are used for training")
    arg_parser.add_argument('--data-time-range', type=interv_list, default="0,2",
                            help="Comma separated list of ensembles that are used for training")
    arg_parser.add_argument('--skip-layers', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--spectral-norm', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--disable-first-bn', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--lstm', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--split-timesteps', type=int, default=1, help="Number of channels for each conditioned label")
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/',
                            help="Parent directory of the training checkpoints and the snapshot images")
    arg_parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                            help="Load all the arguments from a text file")
    arg_parser.add_argument('--test', action='store_true', help="Use image test data set for training")
    arg_parser.add_argument('--mini-batch-discr-dim', type=int, default=None, help="Use image test data set for training")

    return arg_parser


def set_train_args(arg_file=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--n-images', type=int, default=5, help="Number of images to generate for evaluation")
    arg_parser.add_argument('--resume-iter', type=int, help="Iteration step from which the training will be resumed")
    arg_parser.add_argument('--batch-size', type=int, default=18, help="Batch size")
    arg_parser.add_argument('--n-threads', type=int, default=64, help="Number of threads")
    arg_parser.add_argument('--multi-gpus', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    arg_parser.add_argument('--dropout', type=float, default=None, help="Learning rate")
    arg_parser.add_argument('--max-iter', type=int, default=1000000, help="Maximum number of iterations")
    arg_parser.add_argument('--log-interval', type=int, default=None,
                            help="Iteration step interval at which a tensorboard summary log should be written")
    arg_parser.add_argument('--lr-scheduler-patience', type=int, default=None, help="Patience for the lr scheduler")
    arg_parser.add_argument('--save-model-interval', type=int, default=50000,
                            help="Iteration step interval at which the model should be saved")
    arg_parser.add_argument('--save-snapshot-image-interval', type=int, default=None,
                            help="Iteration step interval at which a tensorboard summary log should be written")
    arg_parser.add_argument('--loss', type=str, default="bce",
                            help="bce: binary-cross-entropy loss, "
                                 "hinge: hinge loss, "
                                 "wasserstein: wasserstein loss")
    arg_parser.add_argument('--support-ensemble', type=str, default=None, help="Read data via freva time-frequency")

    global_args(arg_parser, arg_file)


def set_evaluate_args(arg_file=None, prog_func=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--eval-dir', type=str, default='evaluations/',
                            help="Parent directory of the training checkpoints and the snapshot images")
    arg_parser.add_argument('--model-names', type=str_list, default=None,
                            help="Comma separated list of classes for location prediction")
    arg_parser.add_argument('--eval-names', type=str_list, default=None,
                            help="Comma separated list of classes for location prediction")
    arg_parser.add_argument('--val-ensemble', type=str, default=None, help="Read data via freva time-frequency")
    arg_parser.add_argument('--eval-timechunks', type=int_list, default='1,2',
                            help="Tuple of the size of the data (height x width)")
    arg_parser.add_argument('--generate-ensembles', type=int, default=10,
                            help="Iteration step interval at which the model should be saved")
    global_args(arg_parser, arg_file, prog_func)
