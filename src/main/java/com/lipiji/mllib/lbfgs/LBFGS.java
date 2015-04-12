package com.lipiji.mllib.lbfgs;

import org.jblas.DoubleMatrix;

/*
 * Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
 * 
 */
public class LBFGS {

	/*
	 * d: The search direction
	 * x: previous iterate
	 * rho :- The backtrack step between (0,1) usually 1/2
	 * c: parameter between 0 and 1 , usually 10^{-4}
	 * http://www.cnblogs.com/kemaswill/p/3416231.html
	 */
	private static double backtrackLineSearch(Optimizer opt, DoubleMatrix d, double rho, double c) {
		double lambda = 1;
		DoubleMatrix x = opt.getProblem().getTheta();
		int maxIter = 10000;
		while (maxIter > 0 && (opt.getObjectValue(x.add(d.mul(lambda))) >
			opt.getObjectValue(x) + opt.getGradient().transpose().mmul(d).mul(c * lambda).get(0))) {
			lambda *= rho;
			maxIter--;
		}
		return lambda;
	}

	public static DoubleMatrix train(Optimizer opt, int Iter, int m, double e) {
		int dim = opt.getProblem().getTheta().length;
		DoubleMatrix s = new DoubleMatrix(dim, m);
		DoubleMatrix y = s.dup();
		DoubleMatrix rho = new DoubleMatrix(m);
		DoubleMatrix g = opt.getGradient();
		DoubleMatrix x = opt.getProblem().getTheta();
		for (int k = 0; k < Iter; k++) {
			int current = k % m;
			DoubleMatrix d = getHdotg(g, s, y, rho, k, m).mul(-1);
			double lambda = backtrackLineSearch(opt, d, 0.5, 0.0001);
			s.putColumn(current, d.mul(lambda));
			DoubleMatrix newx = x.add(s.getColumn(current));
			opt.update(newx);
			DoubleMatrix newg = opt.getGradient();
			if (newg.norm2() < e) {
				return opt.getProblem().getTheta();
			}
			y.putColumn(current, newg.sub(g));
			double ys = y.getColumn(current).transpose().mmul(s.getColumn(current)).get(0);
			rho.put(current, ys == 0 ? 0 : (1 / ys));
			g = newg.dup();
			x = newx.dup();
			System.out.println(k + " - " + opt.getObjectValue() + " - " + newg.norm2());
		}

		return opt.getProblem().getTheta();
	}

	private static DoubleMatrix getHdotg(DoubleMatrix g, DoubleMatrix s, DoubleMatrix y,
			DoubleMatrix rho, int k, int m) {
		int delta = (k <= m ? 0 : k - m);
		int L = (k <= m ? k : m);
		DoubleMatrix alpha = new DoubleMatrix(L);
		DoubleMatrix beta = alpha.dup();
		DoubleMatrix q = new DoubleMatrix(g.getLength(), L + 1);
		q.putColumn(L, g);
		// backward
		for (int i = L - 1; i >= 0; i--) {
			int j = (i + delta) % m;
			alpha.put(i, s.getColumn(j).transpose().mmul(q.getColumn(i + 1)).mul(rho.get(j)).get(0));
			q.putColumn(i, q.getColumn(i + 1).sub(y.getColumn(j).mul(alpha.get(i))));
		}
		
		// forward
		DoubleMatrix z = q.dup();
		for (int i = 0; i < L; i++) {
			int j = (i + delta) % m;
			beta.put(j, y.getColumn(j).transpose().mmul(z.getColumn(i)).mul(rho.get(j)).get(0));
			z.putColumn(i + 1, z.getColumn(i).add(s.getColumn(j).mul(alpha.get(i) - beta.get(i))));
		}
		return z.getColumn(L);
	}
}
