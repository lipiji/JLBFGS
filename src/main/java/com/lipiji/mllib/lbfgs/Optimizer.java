package com.lipiji.mllib.lbfgs;

import org.jblas.DoubleMatrix;

import com.lipiji.mllib.domain.Problem;

public class Optimizer {
    protected Problem problem;

    public Optimizer(Problem problem) {
        this.problem = problem;
    }

    public void update(DoubleMatrix theta) {
        this.problem.setTheta(theta);
    }

    protected DoubleMatrix getLinearCombination(DoubleMatrix theta) {
        return theta.transpose().mmul(problem.getX()).transpose();
    }

    protected DoubleMatrix getLinearCombination() {
        return getLinearCombination(problem.getTheta());
    }

    // Default: Mean Square Loss
    public double getObjectValue(DoubleMatrix theta) {
        return getLinearCombination(theta).sub(problem.getY()).norm2();
    }

    public double getObjectValue() {
        return getObjectValue(problem.getTheta());
    }

    // Default: Gradient of Mean Square Loss
    public DoubleMatrix getGradient() {
        DoubleMatrix diff = getLinearCombination().sub(problem.getY());
        return problem.getX().mmul(diff).div(problem.getX().getColumns());
    }

    public Problem getProblem() {
        return problem;
    }
}
