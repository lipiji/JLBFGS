package com.lipiji.mmlib.lbfgs;

import com.lipiji.mllib.domain.Problem;
import com.lipiji.mllib.lbfgs.Optimizer;

// Mean Square Loss
public class MSEOptimizer extends Optimizer {
    public MSEOptimizer(Problem problem) {
        super(problem);
    }
}
