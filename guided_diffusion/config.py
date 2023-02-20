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

    global n_classes
    global classes
    if data_root_dir:
        class_names = ["{} {}".format(location, ssi) for location in locations for ssi in train_ssis if
                       (location == 'ne' and ssi == 0.0) or (location != 'ne' and ssi != 0.0)]
        classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        n_classes = len(classes.keys())
    else:
        n_classes = None
        classes = None



    torch.backends.cudnn.benchmark = True
    globals()[device] = torch.device(device)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def set_common_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-root-dir', type=str, default=None,
                            help="Root directory containing the climate datasets")
    arg_parser.add_argument('--log-dir', type=str, default='logs/', help="Directory where the log files will be stored")
    arg_parser.add_argument('--img-names', type=str_list, default='train.nc',
                            help="Comma separated list of netCDF files (climate dataset)")
    arg_parser.add_argument('--img-sizes', type=int_list, default='72,72',
                            help="Tuple of the size of the data (height x width)")
    arg_parser.add_argument('--data-types', type=str_list, default='tas', help="Comma separated list of variable types")
    arg_parser.add_argument('--device', type=str, default='cuda', help="Device used by PyTorch (cuda or cpu)")
    arg_parser.add_argument('--random-seed', type=int, default=None,
                            help="Random seed for iteration loop and initialization weights")
    arg_parser.add_argument('--normalization', type=str, default=None,
                            help="None: No normalization, "
                                 "std: normalize to 0 mean and 1 std, "
                                 "img: normalize values between -1 and 1 and 0.5 mean and 0.5 std, "
                                 "custom: normalize with custom define mean and std values")
    arg_parser.add_argument('--custom-mean', type=float, default=2e-4, help="Custom mean for image normalization")
    arg_parser.add_argument('--custom-std', type=float, default=2e-4, help="Custom std for image normalization")
    arg_parser.add_argument('--n-residual-blocks', type=int, default=2, help="Number of layers")
    arg_parser.add_argument('--diffusion-steps', type=int, default=1000, help="Number of layers")
    arg_parser.add_argument('--conv-factors', type=int_list, default="1,2,3,4",
                            help="Factor of convolutions for increasing model complexity")
    arg_parser.add_argument('--num-channels', type=int, default=128,
                            help="Factor of convolutions for increasing model complexity")
    arg_parser.add_argument('--attention-res', type=int_list, default="16,8",
                            help="Resolutions of current tensors where attention layers should apply")
    arg_parser.add_argument('--freva-project', type=str, default=None, help="Read data via freva project")
    arg_parser.add_argument('--freva-model', type=str, default=None, help="Read data via freva model")
    arg_parser.add_argument('--freva-experiment', type=str, default=None, help="Read data via freva experiment")
    arg_parser.add_argument('--freva-time-frequency', type=str, default=None, help="Read data via freva time-frequency")
    arg_parser.add_argument('-v', '--vlim', type=int_list, default=None,
                            help="Comma separated list of integers defining the number of classes of different labels")
    arg_parser.add_argument('--gt-ensembles', type=str_list, default=None,
                            help="Comma separated list of classes for location prediction")
    arg_parser.add_argument('--train-ssis', type=float_list, default=None,
                            help="Comma separated list of ssi values that are used for training")
    arg_parser.add_argument('--split-timesteps', type=int, default=1, help="Number of channels for each conditioned label")
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/',
                            help="Parent directory of the training checkpoints and the snapshot images")
    arg_parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                            help="Load all the arguments from a text file")
    arg_parser.add_argument('--lstm', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--locations', type=str_list, default=',nh,sh,ne',
                            help="Comma separated list of classes for location prediction")
    arg_parser.add_argument('--mean-input', action='store_true', help="Use a custom padding for global dataset")

    return arg_parser


def set_train_args(arg_file=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--n-images', type=int, default=5, help="Number of images to generate for evaluation")
    arg_parser.add_argument('--resume-iter', type=str, default="", help="Iteration step from which the training will be resumed")
    arg_parser.add_argument('--batch-size', type=int, default=18, help="Batch size")
    arg_parser.add_argument('--n-threads', type=int, default=64, help="Number of threads")
    arg_parser.add_argument('--multi-gpus', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    arg_parser.add_argument('--dropout', type=float, default=0.0, help="Learning rate")
    arg_parser.add_argument('--max-iter', type=int, default=1000000, help="Maximum number of iterations")
    arg_parser.add_argument('--log-interval', type=int, default=None,
                            help="Iteration step interval at which a tensorboard summary log should be written")
    arg_parser.add_argument('--save-model-interval', type=int, default=50000,
                            help="Iteration step interval at which the model should be saved")
    arg_parser.add_argument('--save-snapshot-image-interval', type=int, default=None,
                            help="Iteration step interval at which a tensorboard summary log should be written")
    arg_parser.add_argument('--support-ensemble', type=str, default=None, help="Read data via freva time-frequency")
    arg_parser.add_argument('--train-ensembles', type=interv_list, default='101',
                            help="Comma separated list of ensembles that are used for training")
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
