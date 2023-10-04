import argparse
import numpy as np
from utils import load_data, data_distribution, create_network
from algorithms import *

# ----------------------------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# ----------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Matrix Factorization Problem')
parser.add_argument('--rank', default=50, type=int, help='columns of matrix')
parser.add_argument('--num_workers', default=50, type=int, help='number of workers')
parser.add_argument('--network', default='ring', type=str, help='type of network topology')
parser.add_argument('--distribution', default='random', type=str, help='type of data distribution')
parser.add_argument('--algorithm', default='PDST', type=str, help='name of algorithm')
parser.add_argument('--out_fname', default='', type=str, help='name of output file')
parser.add_argument('--init', default=0.0001, type=float, help='initialization point')
parser.add_argument('--num_iters', default=100, type=int, help='number of iterations')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--beta', default=0.1, type=float, help='parameter of STORM optimizer')
parser.add_argument('--q', default=100, type=int, help='parameter of SPIDER optimizer')
parser.add_argument('--threshold', default=0.002, type=float, help='threshold to adopt perturbation')
parser.add_argument('--radius', default=0.01, type=float, help='perturbation radius')
parser.add_argument('--distance', default=0.5, type=float, help='distance used to discriminate saddle point')
parser.add_argument('--print_freq', default=10, type=int, help='frequency to print stats')
# ----------------------------------------------------------------------------------------------- #


def main():
    args = parser.parse_args()
    data = load_data()
    n, l = data.shape
    # The index of data on i-th worker node is from [intervals[i], intervals[i + 1]).
    intervals = data_distribution(n, args.num_workers, args.distribution)
    W = create_network(args.num_workers, args.network)

    if not args.out_fname:
        args.out_fname = args.algorithm + '.csv'

    if args.algorithm == 'DPSGD':
        DPSGD(args, data, intervals, W)
    elif args.algorithm == 'GTDSGD':
        GTDSGD(args, data, intervals, W)
    elif args.algorithm == 'D-GET':
        DGET(args, data, intervals, W)
    elif args.algorithm == 'D-SPIDER-SFO':
        DSPIDER(args, data, intervals, W)
    elif args.algorithm == 'GTHSGD':
        GTHSGD(args, data, intervals, W)
    elif args.algorithm == 'PDST':
        PDST(args, data, intervals, W)
    else:
        raise NotImplementedError('Not Implemented Yet!') 


if __name__ == '__main__':
    main()
