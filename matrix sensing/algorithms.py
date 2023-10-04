import numpy as np
import os
from utils import get_fval, get_distance, get_gradient, consensus
import pdb


def DPSGD(args, n, M, A, b0, x0, intervals, W):
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    models = []
    for i in range(args.num_workers):
        x = np.copy(x0)
        models.append(np.copy(x))
    # Convert into Ndarray. The shape of models is (nw, d, r).
    models = np.asarray(models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(x, A, b0)
    dist = get_distance(x, M)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval,distance\n')
        f.write('0,0.00,%.4f,%.4f\n' % (fval, dist))
    
    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            batch = np.random.randint(intervals[worker], intervals[worker + 1], args.batch_size)
            gx = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
            models[worker, :, :] = models[worker, :, :] - args.lr * gx
            oracle += len(batch)
        models = consensus(models, W)

        if (1 + iteration) % args.print_freq == 0:
            x = models.mean(axis=0)
            fval = get_fval(x, A, b0)
            dist = get_distance(x, M)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f,%.4f\n' % (1 + iteration, oracle / n, fval, dist))
    
    return models.mean(axis=0)


def GTDSGD(args, n, M, A, b0, x0, intervals, W):
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    models = []
    for i in range(args.num_workers):
        x = np.copy(x0)
        models.append(np.copy(x))
    # Convert into Ndarray. The shape of models is (nw, d, r).
    models = np.asarray(models)
    gradients = np.zeros_like(models)
    gradient_buffers = np.zeros_like(models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(x, A, b0)
    dist = get_distance(x, M)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval,distance\n')
        f.write('0,0.00,%.4f,%.4f\n' % (fval, dist))

    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            batch = np.random.randint(intervals[worker], intervals[worker + 1], args.batch_size)
            gx = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
            gradients[worker, :, :] = gradients[worker, :, :] + gx - gradient_buffers[worker, :, :]
            gradient_buffers[worker, :, :] = gx
            oracle += len(batch)
        gradients = consensus(gradients, W)
        models = models - args.lr * gradients
        models = consensus(models, W)

        if (1 + iteration) % args.print_freq == 0:
            x = models.mean(axis=0)
            fval = get_fval(x, A, b0)
            dist = get_distance(x, M)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f,%.4f\n' % (1 + iteration, oracle / n, fval, dist))

    return models.mean(axis=0)
    

def DGET(args, n, M, A, b0, x0, intervals, W):
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    models = []
    for i in range(args.num_workers):
        x = np.copy(x0)
        models.append(np.copy(x))
    # Convert into Ndarray. The shape of models is (nw, d, r).
    models = np.asarray(models)
    model_buffers = np.copy(models)
    gradients = np.zeros_like(models)
    gradient_buffers = np.zeros_like(models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(x, A, b0)
    dist = get_distance(x, M)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval,distance\n')
        f.write('0,0.00,%.4f,%.4f\n' % (fval, dist))

    # Start training
    for iteration in range(args.num_iters):
        model_buffers, models = models, consensus(models, W)
        models = models - args.lr * gradients
        gradients = consensus(gradients, W)
        for worker in range(args.num_workers):
            if iteration % args.q == 0:
                batch = np.array(range(intervals[worker], intervals[worker + 1])) 
                vx = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
            else:
                batch = np.random.randint(intervals[worker], intervals[worker + 1], args.batch_size)
                gx = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
                gx_old = get_gradient(model_buffers[worker, :, :], A[:, :, batch], b0[batch])
                vx = gx + gradient_buffers[worker, :, :] - gx_old
            oracle += len(batch)
            gradients[worker, :, :] = gradients[worker, :, :] + vx - gradient_buffers[worker, :, :]
            gradient_buffers[worker, :, :] = vx
        
        if (1 + iteration) % args.print_freq == 0:
            x = models.mean(axis=0)
            fval = get_fval(x, A, b0)
            dist = get_distance(x, M)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f,%.4f\n' % (1 + iteration, oracle / n, fval, dist))

    return models.mean(axis=0)


def DSPIDER(args, n, M, A, b0, x0, intervals, W):
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    models = []
    for i in range(args.num_workers):
        x = np.copy(x0)
        models.append(np.copy(x))
    # Convert into Ndarray. The shape of models is (nw, d, r).
    models = np.asarray(models)
    model_buffers = np.copy(models)
    gradients = np.zeros_like(models)
    gradient_buffers = np.zeros_like(models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(x, A, b0)
    dist = get_distance(x, M)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval,distance\n')
        f.write('0,0.00,%.4f,%.4f\n' % (fval, dist))

    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            if iteration % args.q == 0:
                batch = np.array(range(intervals[worker], intervals[worker + 1])) 
                gradients[worker, :, :] = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
            else:
                batch = np.random.randint(intervals[worker], intervals[worker + 1], args.batch_size)
                gx = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
                gx_old = get_gradient(model_buffers[worker, :, :], A[:, :, batch], b0[batch])
                gradients[worker, :, :] = gx + gradient_buffers[worker, :, :] - gx_old
            oracle += len(batch)
        gamma = 0.25 # The value of gamma can be changed. In the paper of D-SPIDER-SFO gamma = 1.
        model_buffers, models = models, models - args.lr * gradients + gamma * (models - model_buffers + args.lr * gradient_buffers) 
        models = consensus(models, W)
        gradient_buffers = gradients.copy()
        
        if (1 + iteration) % args.print_freq == 0:
            x = models.mean(axis=0)
            fval = get_fval(x, A, b0)
            dist = get_distance(x, M)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f,%.4f\n' % (1 + iteration, oracle / n, fval, dist))

    return models.mean(axis=0)


def GTHSGD(args, n, M, A, b0, x0, intervals, W):
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    models = []
    for i in range(args.num_workers):
        x = np.copy(x0)
        models.append(np.copy(x))
    # Convert into Ndarray. The shape of models is (nw, d, r).
    models = np.asarray(models)
    model_buffers = np.zeros_like(models)
    gradients = np.zeros_like(models)
    gradient_buffers = np.zeros_like(models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(x, A, b0)
    dist = get_distance(x, M)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval,distance\n')
        f.write('0,0.00,%.4f,%.4f\n' % (fval, dist))

    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            if iteration == 0:
                batch = np.array(range(intervals[worker], intervals[worker + 1])) 
                gradient_buffers[worker, :, :] = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
                gradients[worker, :, :] = gradient_buffers[worker, :, :].copy()
            else:
                batch = np.random.randint(intervals[worker], intervals[worker + 1], args.batch_size)
                gx = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
                gx_old = get_gradient(model_buffers[worker, :, :], A[:, :, batch], b0[batch])
                vx = gx + (1 - args.beta) * (gradient_buffers[worker, :, :] - gx_old)
                gradients[worker, :, :] = gradients[worker, :, :] + vx - gradient_buffers[worker, :, :]
                gradient_buffers[worker, :, :] = vx
            oracle += len(batch)
        gradients = consensus(gradients, W)
        model_buffers, models = models, models - args.lr * gradients
        models = consensus(models, W)

        if (1 + iteration) % args.print_freq == 0:
            x = models.mean(axis=0)
            fval = get_fval(x, A, b0)
            dist = get_distance(x, M)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f,%.4f\n' % (1 + iteration, oracle / n, fval, dist))

    return models.mean(axis=0)


def PDST(args, n, M, A, b0, x0, intervals, W):
    # Before consensus, buffers save temporary values. After each iteration, buffers save old values.
    models = []
    anchors = []
    for i in range(args.num_workers):
        x = np.copy(x0)
        models.append(np.copy(x))
        anchors.append(np.copy(x))
    # Convert into Ndarray. The shape of models is (nw, d, r).
    models = np.asarray(models)
    model_buffers = np.zeros_like(models)
    gradients = np.zeros_like(models)
    gradient_buffers = np.zeros_like(models)

    out_fname = os.path.join('result', args.out_fname)
    oracle = 0
    fval = get_fval(x, A, b0)
    dist = get_distance(x, M)
    with open(out_fname, 'w') as f:
        f.write('iteration,oracle,fval,distance\n')
        f.write('0,0.00,%.4f,%.4f\n' % (fval, dist))

    perturbation = [False] * args.num_workers
    escaping = [False] * args.num_workers

    # Start training
    for iteration in range(args.num_iters):
        for worker in range(args.num_workers):
            if iteration == 0 or perturbation[worker]:
                batch = np.array(range(intervals[worker], intervals[worker + 1])) 
                gradient_buffers[worker, :, :] = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
                gradients[worker, :, :] = gradient_buffers[worker, :, :].copy()
                perturbation[worker] = False
            else:
                batch = np.random.randint(intervals[worker], intervals[worker + 1], args.batch_size)
                gx = get_gradient(models[worker, :, :], A[:, :, batch], b0[batch])
                gx_old = get_gradient(model_buffers[worker, :, :], A[:, :, batch], b0[batch])
                vx = gx + (1 - args.beta) * (gradient_buffers[worker, :, :] - gx_old)
                gradients[worker, :, :] = gradients[worker, :, :] + vx - gradient_buffers[worker, :, :]
                gradient_buffers[worker, :, :] = vx
            oracle += len(batch)
        gradients = consensus(gradients, W)
        for worker in range(args.num_workers):
            if not escaping[worker] and np.sqrt(np.sum(gradients[worker, :, :] * gradients[worker, :, :])) <= args.threshold:
                direction = np.random.rand(args.dim, args.rank)
                direction = direction / np.sqrt(np.sum(direction * direction))
                model_buffers[worker, :, :] = models[worker, :, :] + np.random.rand() * args.radius * direction
                perturbation[worker], escaping[worker] = True, True
                anchors[worker] = models[worker, :, :].copy()
            else:
                model_buffers[worker, :, :] = models[worker, :, :] - args.lr * gradients[worker, :, :]
        model_buffers, models = models, consensus(model_buffers, W)
        for worker in range(args.num_workers):
            if escaping[worker] and np.sqrt(np.sum((models[worker, :, :] - anchors[worker]) * (models[worker, :, :] - anchors[worker]))) >= args.distance:
                escaping[worker] = False

        if (1 + iteration) % args.print_freq == 0:
            x = models.mean(axis=0)
            fval = get_fval(x, A, b0)
            dist = get_distance(x, M)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%.4f,%.4f\n' % (1 + iteration, oracle / n, fval, dist))

    return models.mean(axis=0)
