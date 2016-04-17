package com.lipiji.mllib.lbfgs;

import org.jblas.DoubleMatrix;

/*
 * Broyden-Fletcher-Goldfarb-Shanno (BFGS)
 */
public class BFGS {

    /*
    * d: The search direction
    * x: previous iterate
    * rho :- The backtrack step between (0,1) usually 1/2
    * c: parameter between 0 and 1 , usually 10^{-4}
    * http://www.cnblogs.com/kemaswill/p/3416231.html
    */
    protected static double backtrackLineSearch(Optimizer opt, DoubleMatrix d,
            double rho, double c) {
        double lambda = 1;
        DoubleMatrix x = opt.getProblem().getTheta();
        int maxIter = 10000;
        while (maxIter > 0 && (opt.getObjectValue(x.add(d.mul(lambda)))
                        > opt.getObjectValue(x) + opt.getGradient().transpose().mmul(d).mul(c * lambda).get(0))) {
            lambda *= rho;
            maxIter--;
        }
        return lambda;
    }

    public static DoubleMatrix train(Optimizer opt, int Iter, double e) {
        int dim = opt.getProblem().getTheta().length;
        DoubleMatrix s = new DoubleMatrix(dim, 1);
        DoubleMatrix y = s.dup();
        DoubleMatrix I = DoubleMatrix.eye(dim);
        DoubleMatrix D = I.dup();
        DoubleMatrix g = opt.getGradient();
        DoubleMatrix x = opt.getProblem().getTheta();
        for (int k = 0; k < Iter; k++) {
            DoubleMatrix d = D.mmul(g).mmul(-1);
            double lambda = backtrackLineSearch(opt, d, 0.5, 0.0001);
            s = d.mmul(lambda);
            x = x.add(s);
            opt.update(x);
            DoubleMatrix newg = opt.getGradient();
            y = newg.sub(g);
            if (y.norm2() < e) {
                return opt.getProblem().getTheta();
            }
            g = newg.dup();

            double ys = y.transpose().mmul(s).get(0);
            D = (I.sub(s.mmul(y.transpose()).div(ys))).mmul(D)
                    .mmul((I.sub(y.mmul(s.transpose()).div(ys))))
                    .add(s.mmul(s.transpose()).div(ys));

            System.out.println(k + " - " + opt.getObjectValue() + " - " + g.norm2());
        }
        return opt.getProblem().getTheta();
    }
}
