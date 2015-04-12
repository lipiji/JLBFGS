package com.lipiji.mmlib.lbfgs;

import org.jblas.DoubleMatrix;

import com.lipiji.mllib.domain.Problem;
import com.lipiji.mllib.lbfgs.Optimizer;

// Mean Square Loss
public class LogisticRegressionOptimizer extends Optimizer {
	public LogisticRegressionOptimizer(Problem problem) {
		super(problem);
	}
	
	private double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	@Override
	public double getObjectValue(DoubleMatrix theta) {
		DoubleMatrix h = getLinearCombination(theta);
		for (int i = 0; i < h.length; i++) {
			h.put(i, sigmoid(h.get(i)));
		}
		return h.sub(problem.getY()).norm2();
	}

	@Override
	public double getObjectValue() {
		return getObjectValue(problem.getTheta());
	}

	@Override
	public DoubleMatrix getGradient() {
		DoubleMatrix h = getLinearCombination();
		for (int i = 0; i < h.length; i++) {
			h.put(i, sigmoid(h.get(i)));
		}
		DoubleMatrix diff = h.sub(problem.getY());
		return problem.getX().mmul(diff).div(problem.getX().getColumns());
	}
}
