import argparse
from utils import create_data, initialization, data_distribution, create_network, get_hessian_eig
from algorithms import *

# ----------------------------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# ----------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Matrix Sensing Problem')
parser.add_argument('--dim', default=50, type=int, help='rows of matrix')
parser.add_argument('--rank', default=3, type=int, help='columns of matrix')
parser.add_argument('--num_workers', default=20, type=int, help='number of workers')
parser.add_argument('--network', default='ring', type=str, help='type of network topology')
parser.add_argument('--distribution', default='random', type=str, help='type of data distribution')
parser.add_argument('--algorithm', default='PDST', type=str, help='name of algorithm')
parser.add_argument('--out_fname', default='', type=str, help='name of output file')
parser.add_argument('--compare', action='store_true', help='using the same initialization')
parser.add_argument('--num_iters', default=100, type=int, help='number of iterations')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--beta', default=0.01, type=float, help='parameter of STORM optimizer')
parser.add_argument('--q', default=100, type=int, help='parameter of SPIDER optimizer')
parser.add_argument('--threshold', default=0.0001, type=float, help='threshold to adopt perturbation')
parser.add_argument('--radius', default=0.001, type=float, help='perturbation radius')
parser.add_argument('--distance', default=0.01, type=float, help='distance used to discriminate saddle point')
parser.add_argument('--print_freq', default=10, type=int, help='frequency to print stats')
# ----------------------------------------------------------------------------------------------- #


def main():
    args = parser.parse_args()
    n = 20 * args.dim * args.num_workers
    # Create data. M is dxd, A is dxdxn
    M, A, b0 = create_data(args.dim, args.rank, n)
    # Initialization. x is a dxr matrix
    x0 = initialization(args.dim, args.rank, M, args.compare)
    # The index of data on i-th worker node is from [intervals[i], intervals[i + 1]).
    intervals = data_distribution(n, args.num_workers, args.distribution)
    W = create_network(args.num_workers, args.network)

    if not args.out_fname:
        args.out_fname = args.algorithm + '.csv'

    if args.algorithm == 'DPSGD':
        x = DPSGD(args, n, M, A, b0, x0, intervals, W)
    elif args.algorithm == 'GTDSGD':
        x = GTDSGD(args, n, M, A, b0, x0, intervals, W)
    elif args.algorithm == 'D-GET':
        x = DGET(args, n, M, A, b0, x0, intervals, W)
    elif args.algorithm == 'D-SPIDER-SFO':
        x = DSPIDER(args, n, M, A, b0, x0, intervals, W)
    elif args.algorithm == 'GTHSGD':
        x = GTHSGD(args, n, M, A, b0, x0, intervals, W)
    elif args.algorithm == 'PDST':
        x = PDST(args, n, M, A, b0, x0, intervals, W)
    else:
        raise NotImplementedError('Not Implemented Yet!') 

    print(get_hessian_eig(x, A, b0))


if __name__ == '__main__':
    main()
