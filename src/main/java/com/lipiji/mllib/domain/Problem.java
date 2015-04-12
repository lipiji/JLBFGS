package com.lipiji.mllib.domain;

import org.jblas.DoubleMatrix;

public class Problem {
	private DoubleMatrix X;
	private DoubleMatrix Y;
	private DoubleMatrix theta;

	public Problem(DoubleMatrix X, DoubleMatrix Y, DoubleMatrix theta) {
		this.X = X;
		this.Y = Y;
		this.theta = theta;
	}

	public DoubleMatrix getX() {
		return X;
	}

	public void setX(DoubleMatrix x) {
		X = x;
	}

	public DoubleMatrix getY() {
		return Y;
	}

	public void setY(DoubleMatrix y) {
		Y = y;
	}

	public DoubleMatrix getTheta() {
		return theta;
	}

	public void setTheta(DoubleMatrix theta) {
		this.theta = theta;
	}
}
