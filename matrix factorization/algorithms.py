import numpy as np
import os
from utils import get_fval, get_gradient, consensus
import pdb


def DPSGD(args, data, intervals, W):
    # Initialization
    n, l = data.shape
    xu = args.init * np.ones((n, args.rank))
    xv = args.init * np.ones((l, args.rank))
    # Data partition
    slices = []
    for worker in range(args.num_workers):
        slices.append(np.array(range(intervals[worker], intervals[worker + 1])))
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    u_models = []
    v_models = []
    for i in range(args.num_workers):
        u_models.append(np.copy(xu))
        v_models.append(np.copy(xv))
    # Convert into Ndarray. The shape of u_models is (nw, n, r). The shape of v_models is (nw, l, r).
    u_models = np.asarray(u_models)
    v_models = np.asarray(v_models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(xu, xv, data)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval\n')
        f.write('0,0.00,%.4f\n' % (fval))
    
    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            batch = np.random.randint(0, l, args.batch_size)
            gu, gv = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
            u_models[worker, slices[worker], :] = u_models[worker, slices[worker], :] - args.lr * gu
            v_models[worker, batch, :] = v_models[worker, batch, :] - args.lr * gv
            oracle += len(batch)
        u_models = consensus(u_models, W)
        v_models = consensus(v_models, W)

        if (1 + iteration) % args.print_freq == 0:
            xu = u_models.mean(axis=0)
            xv = v_models.mean(axis=0)
            fval = get_fval(xu, xv, data)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f\n' % (1 + iteration, oracle / (n * l), fval))
    

def GTDSGD(args, data, intervals, W):
    # Initialization
    n, l = data.shape
    xu = args.init * np.ones((n, args.rank))
    xv = args.init * np.ones((l, args.rank))
    # Data partition
    slices = []
    for worker in range(args.num_workers):
        slices.append(np.array(range(intervals[worker], intervals[worker + 1])))
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    u_models = []
    v_models = []
    for i in range(args.num_workers):
        u_models.append(np.copy(xu))
        v_models.append(np.copy(xv))
    # Convert into Ndarray. The shape of u_models is (nw, n, r). The shape of v_models is (nw, l, r).
    u_models = np.asarray(u_models)
    v_models = np.asarray(v_models)
    u_gradients = np.zeros_like(u_models)
    u_gradient_buffers = np.zeros_like(u_models)
    v_gradients = np.zeros_like(v_models)
    v_gradient_buffers = np.zeros_like(v_models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(xu, xv, data)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval\n')
        f.write('0,0.00,%.4f\n' % (fval))
    
    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            u_temp, v_temp = np.zeros_like(xu), np.zeros_like(xv)
            batch = np.random.randint(0, l, args.batch_size)
            gu, gv = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
            u_temp[slices[worker], :] = gu
            v_temp[batch, :] = gv
            u_gradients[worker, :, :] = u_gradients[worker, :, :] + u_temp - u_gradient_buffers[worker, :, :]
            u_gradient_buffers[worker, :, :] = u_temp
            v_gradients[worker, :, :] = v_gradients[worker, :, :] + v_temp - v_gradient_buffers[worker, :, :]
            v_gradient_buffers[worker, :, :] = v_temp
            oracle += len(batch)
        u_gradients = consensus(u_gradients, W)
        v_gradients = consensus(v_gradients, W)
        u_models = u_models - args.lr * u_gradients
        v_models = v_models - args.lr * v_gradients
        u_models = consensus(u_models, W)
        v_models = consensus(v_models, W)

        if (1 + iteration) % args.print_freq == 0:
            xu = u_models.mean(axis=0)
            xv = v_models.mean(axis=0)
            fval = get_fval(xu, xv, data)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f\n' % (1 + iteration, oracle / (n * l), fval))

    
def DGET(args, data, intervals, W):
    # Initialization
    n, l = data.shape
    xu = args.init * np.ones((n, args.rank))
    xv = args.init * np.ones((l, args.rank))
    # Data partition
    slices = []
    for worker in range(args.num_workers):
        slices.append(np.array(range(intervals[worker], intervals[worker + 1])))
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    u_models = []
    v_models = []
    for i in range(args.num_workers):
        u_models.append(np.copy(xu))
        v_models.append(np.copy(xv))
    # Convert into Ndarray. The shape of u_models is (nw, n, r). The shape of v_models is (nw, l, r).
    u_models = np.asarray(u_models)
    v_models = np.asarray(v_models)
    u_model_buffers = np.copy(u_models)
    v_model_buffers = np.copy(v_models)
    u_gradients = np.zeros_like(u_models)
    u_gradient_buffers = np.zeros_like(u_models)
    v_gradients = np.zeros_like(v_models)
    v_gradient_buffers = np.zeros_like(v_models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(xu, xv, data)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval\n')
        f.write('0,0.00,%.4f\n' % (fval))
    
    # Start training
    for iteration in range(args.num_iters):
        u_model_buffers, u_models = u_models, consensus(u_models, W)
        v_model_buffers, v_models = v_models, consensus(v_models, W)
        u_models = u_models - args.lr * u_gradients
        v_models = v_models - args.lr * v_gradients
        u_gradients = consensus(u_gradients, W)
        v_gradients = consensus(v_gradients, W)

        for worker in range(args.num_workers):
            if iteration % args.q == 0:
                u_temp = np.zeros_like(xu)
                batch = np.array(range(l)) 
                gu, v_temp = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
                u_temp[slices[worker], :] = gu
            else:
                u_temp = np.copy(u_gradient_buffers[worker, :, :])
                v_temp = np.copy(v_gradient_buffers[worker, :, :])
                batch = np.random.randint(0, l, args.batch_size)
                gu, gv = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
                gu_old, gv_old = get_gradient(u_model_buffers[worker, slices[worker], :], v_model_buffers[worker, batch, :], data[slices[worker][:, None], batch])
                u_temp[slices[worker], :] = gu + u_temp[slices[worker], :] - gu_old
                v_temp[batch, :] = gv + v_temp[batch, :] - gv_old
            oracle += len(batch)
            u_gradients[worker, :, :] = u_gradients[worker, :, :] + u_temp - u_gradient_buffers[worker, :, :]
            u_gradient_buffers[worker, :, :] = u_temp
            v_gradients[worker, :, :] = v_gradients[worker, :, :] + v_temp - v_gradient_buffers[worker, :, :]
            v_gradient_buffers[worker, :, :] = v_temp
        
        if (1 + iteration) % args.print_freq == 0:
            xu = u_models.mean(axis=0)
            xv = v_models.mean(axis=0)
            fval = get_fval(xu, xv, data)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f\n' % (1 + iteration, oracle / (n * l), fval))


def DSPIDER(args, data, intervals, W):
    # Initialization
    n, l = data.shape
    xu = args.init * np.ones((n, args.rank))
    xv = args.init * np.ones((l, args.rank))
    # Data partition
    slices = []
    for worker in range(args.num_workers):
        slices.append(np.array(range(intervals[worker], intervals[worker + 1])))
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    u_models = []
    v_models = []
    for i in range(args.num_workers):
        u_models.append(np.copy(xu))
        v_models.append(np.copy(xv))
    # Convert into Ndarray. The shape of u_models is (nw, n, r). The shape of v_models is (nw, l, r).
    u_models = np.asarray(u_models)
    v_models = np.asarray(v_models)
    u_model_buffers = np.copy(u_models)
    v_model_buffers = np.copy(v_models)
    u_gradients = np.zeros_like(u_models)
    u_gradient_buffers = np.zeros_like(u_models)
    v_gradients = np.zeros_like(v_models)
    v_gradient_buffers = np.zeros_like(v_models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(xu, xv, data)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval\n')
        f.write('0,0.00,%.4f\n' % (fval))

    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            if iteration % args.q == 0:
                u_temp = np.zeros_like(xu)
                batch = np.array(range(l)) 
                gu, v_temp = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
                u_temp[slices[worker], :] = gu
                u_gradients[worker, :, :], v_gradients[worker, :, :] = u_temp, v_temp
            else:
                u_temp = np.copy(u_gradient_buffers[worker, :, :])
                v_temp = np.copy(v_gradient_buffers[worker, :, :])
                batch = np.random.randint(0, l, args.batch_size)
                gu, gv = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
                gu_old, gv_old = get_gradient(u_model_buffers[worker, slices[worker], :], v_model_buffers[worker, batch, :], data[slices[worker][:, None], batch])
                u_temp[slices[worker], :] = gu + u_temp[slices[worker], :] - gu_old
                v_temp[batch, :] = gv + v_temp[batch, :] - gv_old
                u_gradients[worker, :, :], v_gradients[worker, :, :] = u_temp, v_temp
            oracle += len(batch)
        gamma = 0.25 # The value of gamma can be changed. In the paper of D-SPIDER-SFO gamma = 1.
        u_model_buffers, u_models = u_models, u_models - args.lr * u_gradients + gamma * (u_models - u_model_buffers + args.lr * u_gradient_buffers) 
        v_model_buffers, v_models = v_models, v_models - args.lr * v_gradients + gamma * (v_models - v_model_buffers + args.lr * v_gradient_buffers) 
        u_models = consensus(u_models, W)
        v_models = consensus(v_models, W)
        u_gradient_buffers = u_gradients.copy()
        v_gradient_buffers = v_gradients.copy()
        
        if (1 + iteration) % args.print_freq == 0:
            xu = u_models.mean(axis=0)
            xv = v_models.mean(axis=0)
            fval = get_fval(xu, xv, data)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f\n' % (1 + iteration, oracle / (n * l), fval))


def GTHSGD(args, data, intervals, W):
    # Initialization
    n, l = data.shape
    xu = args.init * np.ones((n, args.rank))
    xv = args.init * np.ones((l, args.rank))
    # Data partition
    slices = []
    for worker in range(args.num_workers):
        slices.append(np.array(range(intervals[worker], intervals[worker + 1])))
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    u_models = []
    v_models = []
    for i in range(args.num_workers):
        u_models.append(np.copy(xu))
        v_models.append(np.copy(xv))
    # Convert into Ndarray. The shape of u_models is (nw, n, r). The shape of v_models is (nw, l, r).
    u_models = np.asarray(u_models)
    v_models = np.asarray(v_models)
    u_model_buffers = np.copy(u_models)
    v_model_buffers = np.copy(v_models)
    u_gradients = np.zeros_like(u_models)
    u_gradient_buffers = np.zeros_like(u_models)
    v_gradients = np.zeros_like(v_models)
    v_gradient_buffers = np.zeros_like(v_models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(xu, xv, data)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval\n')
        f.write('0,0.00,%.4f\n' % (fval))

    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            if iteration == 0:
                u_temp = np.zeros_like(xu)
                batch = np.array(range(l)) 
                gu, v_temp = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
                u_temp[slices[worker], :] = gu
                u_gradients[worker, :, :], v_gradients[worker, :, :] = u_temp.copy(), v_temp.copy()
                u_gradient_buffers[worker, :, :], v_gradient_buffers[worker, :, :] = u_temp.copy(), v_temp.copy()
            else:
                u_temp, v_temp = np.zeros_like(xu), np.zeros_like(xv)
                u_temp_old, v_temp_old = np.zeros_like(xu), np.zeros_like(xv)
                batch = np.random.randint(0, l, args.batch_size)
                gu, gv = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
                gu_old, gv_old = get_gradient(u_model_buffers[worker, slices[worker], :], v_model_buffers[worker, batch, :], data[slices[worker][:, None], batch])
                u_temp[slices[worker], :], u_temp_old[slices[worker], :] = gu, gu_old
                v_temp[batch, :], v_temp_old[batch, :] = gv, gv_old
                gu_new = u_temp + (1 - args.beta) * (u_gradient_buffers[worker, :, :] - u_temp_old)
                gv_new = v_temp + (1 - args.beta) * (v_gradient_buffers[worker, :, :] - v_temp_old)
                u_gradients[worker, :, :] = u_gradients[worker, :, :] + gu_new - u_gradient_buffers[worker, :, :]
                v_gradients[worker, :, :] = v_gradients[worker, :, :] + gv_new - v_gradient_buffers[worker, :, :]
                u_gradient_buffers[worker, :, :] = gu_new
                v_gradient_buffers[worker, :, :] = gv_new
            oracle += len(batch)
        u_gradients = consensus(u_gradients, W)
        v_gradients = consensus(v_gradients, W)
        u_model_buffers, u_models = u_models, u_models - args.lr * u_gradients
        v_model_buffers, v_models = v_models, v_models - args.lr * v_gradients
        u_models = consensus(u_models, W)
        v_models = consensus(v_models, W)

        if (1 + iteration) % args.print_freq == 0:
            xu = u_models.mean(axis=0)
            xv = v_models.mean(axis=0)
            fval = get_fval(xu, xv, data)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f\n' % (1 + iteration, oracle / (n * l), fval))


def PDST(args, data, intervals, W):
    # Initialization
    n, l = data.shape
    xu = args.init * np.ones((n, args.rank))
    xv = args.init * np.ones((l, args.rank))
    # Data partition
    slices = []
    for worker in range(args.num_workers):
        slices.append(np.array(range(intervals[worker], intervals[worker + 1])))
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    u_models = []
    v_models = []
    u_anchors = []
    v_anchors = []
    for i in range(args.num_workers):
        u_models.append(np.copy(xu))
        v_models.append(np.copy(xv))
        u_anchors.append(np.copy(xu))
        v_anchors.append(np.copy(xv))
    # Convert into Ndarray. The shape of u_models is (nw, n, r). The shape of v_models is (nw, l, r).
    u_models = np.asarray(u_models)
    v_models = np.asarray(v_models)
    u_model_buffers = np.copy(u_models)
    v_model_buffers = np.copy(v_models)
    u_gradients = np.zeros_like(u_models)
    u_gradient_buffers = np.zeros_like(u_models)
    v_gradients = np.zeros_like(v_models)
    v_gradient_buffers = np.zeros_like(v_models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(xu, xv, data)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval\n')
        f.write('0,0.00,%.4f\n' % (fval))

    perturbation = [False] * args.num_workers
    escaping = [False] * args.num_workers

    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            if iteration == 0 or perturbation[worker]:
                u_temp = np.zeros_like(xu)
                batch = np.array(range(l)) 
                gu, v_temp = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
                u_temp[slices[worker], :] = gu
                u_gradients[worker, :, :], v_gradients[worker, :, :] = u_temp.copy(), v_temp.copy()
                u_gradient_buffers[worker, :, :], v_gradient_buffers[worker, :, :] = u_temp.copy(), v_temp.copy()
                perturbation[worker] = False
            else:
                u_temp, v_temp = np.zeros_like(xu), np.zeros_like(xv)
                u_temp_old, v_temp_old = np.zeros_like(xu), np.zeros_like(xv)
                batch = np.random.randint(0, l, args.batch_size)
                gu, gv = get_gradient(u_models[worker, slices[worker], :], v_models[worker, batch, :], data[slices[worker][:, None], batch])
                gu_old, gv_old = get_gradient(u_model_buffers[worker, slices[worker], :], v_model_buffers[worker, batch, :], data[slices[worker][:, None], batch])
                u_temp[slices[worker], :], u_temp_old[slices[worker], :] = gu, gu_old
                v_temp[batch, :], v_temp_old[batch, :] = gv, gv_old
                gu_new = u_temp + (1 - args.beta) * (u_gradient_buffers[worker, :, :] - u_temp_old)
                gv_new = v_temp + (1 - args.beta) * (v_gradient_buffers[worker, :, :] - v_temp_old)
                u_gradients[worker, :, :] = u_gradients[worker, :, :] + gu_new - u_gradient_buffers[worker, :, :]
                v_gradients[worker, :, :] = v_gradients[worker, :, :] + gv_new - v_gradient_buffers[worker, :, :]
                u_gradient_buffers[worker, :, :] = gu_new
                v_gradient_buffers[worker, :, :] = gv_new
            oracle += len(batch)
        u_gradients = consensus(u_gradients, W)
        v_gradients = consensus(v_gradients, W)
        for worker in range(args.num_workers):
            if not escaping[worker] and np.sqrt(np.mean(u_gradients[worker, :, :] * u_gradients[worker, :, :]) \
                    + np.mean(v_gradients[worker, :, :] * v_gradients[worker, :, :])) <= args.threshold:
                u_direction = np.random.rand(n, args.rank)
                u_direction = u_direction / np.sqrt(np.sum(u_direction * u_direction))
                v_direction = np.random.rand(l, args.rank)
                v_direction = v_direction / np.sqrt(np.sum(v_direction * v_direction))
                ra = np.random.rand() * args.radius
                u_model_buffers[worker, :, :] = u_models[worker, :, :] + ra * u_direction
                v_model_buffers[worker, :, :] = v_models[worker, :, :] + ra * v_direction
                perturbation[worker], escaping[worker] = True, True
                u_anchors[worker] = u_models[worker, :, :].copy()
                v_anchors[worker] = v_models[worker, :, :].copy()
            else:
                u_model_buffers[worker, :, :] = u_models[worker, :, :] - args.lr * u_gradients[worker, :, :]
                v_model_buffers[worker, :, :] = v_models[worker, :, :] - args.lr * v_gradients[worker, :, :]
        u_model_buffers, u_models = u_models, consensus(u_model_buffers, W)
        v_model_buffers, v_models = v_models, consensus(v_model_buffers, W)
        for worker in range(args.num_workers):
            if escaping[worker] and np.sqrt(np.mean((u_models[worker, :, :] - u_anchors[worker]) * (u_models[worker, :, :] - u_anchors[worker])) \
                    + np.mean((v_models[worker, :, :] - v_anchors[worker]) * (v_models[worker, :, :] - v_anchors[worker]))) >= args.distance:
                escaping[worker] = False

        if (1 + iteration) % args.print_freq == 0:
            xu = u_models.mean(axis=0)
            xv = v_models.mean(axis=0)
            fval = get_fval(xu, xv, data)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f\n' % (1 + iteration, oracle / (n * l), fval))
