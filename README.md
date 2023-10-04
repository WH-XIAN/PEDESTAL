# PEDESTAL

This is the code to reproduce the matrix sensing and matrix factorization experiments in the paper "Finding Local Minima Efficiently in Decentralized Optimization" in NeurIPS 2023.

## Matrix Sensing 
Go to the matrix sensing folder and run the command 
```
python matrix_sensing.py
```

## Matrix Factorization 
Go to the matrix factorization folder and run the command 
```
python matrix_factorization.py
```

## Parameters
The definition of parameters can be found by running
```
python matrix_sensing.py --help
```
or 
```
python matrix_factorization.py --help
```
Here we will introduce some important parameters in these two tasks.
<pre>
  --num_workers      The number of worker nodes.
  --network          The type of network topology. Choices are 'ring', 'toroidal' and 'exponential'
  --distribution     The type of data distribution. Choices are 'random' and 'dirichlet'
  --algorithm        The name of algorithm. Choices are 'DPSGD', 'GTDSGD', 'D-GET', 'D-SPIDER-SFO', 'GTHSGD' and 'PDST'
  --threshold        The threshold of gradient norm to adopt perturbation.
  --radius           The radius of the ball to draw the perturbation.
  --distance         The distance used to discriminate a saddle point.
</pre>
