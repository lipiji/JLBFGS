package com.lipiji.mmlib.lbfgs;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;

import com.lipiji.mllib.domain.Problem;
import com.lipiji.mllib.lbfgs.BFGS;

public class TestBFGS {
	private DoubleMatrix X;
	private DoubleMatrix Y;
	private DoubleMatrix theta;

	@Before
	public void loadData() {
		try {
			List<DoubleMatrix> XList = new ArrayList<>();
			List<Double> YList = new ArrayList<>();
            InputStream is = this.getClass().getClassLoader().getResourceAsStream("heart_scale");
            BufferedReader bf = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
            String line = null;
            while ((line = bf.readLine()) != null) {
                line = line.trim();
                String[] terms = StringUtils.split(line, " ");
				DoubleMatrix xi = new DoubleMatrix(13, 1);
				for (int i = 0; i < terms.length; i++) {
					String term = terms[i];
					if (term.contains(":")) {
						String[] kv = term.split(":");
						xi.put(Integer.parseInt(kv[0]) - 1, Double.parseDouble(kv[1]));
					} else {
						YList.add(Double.valueOf(term));
					}
				}
				XList.add(xi);
            }
            is.close();
            
    		if (XList.size() != YList.size()) {
    			System.err.println("data error.");
    		} else {
    			X = new DoubleMatrix(13, XList.size());
    			theta = new DoubleMatrix(13, 1);
    			Y = new DoubleMatrix(YList.size(), 1);
    			for (int i = 0; i < YList.size(); i++) {
    				X.putColumn(i, XList.get(i));
    				double label = YList.get(i);
    				label = (label == -1 ? 0 : label);
    				Y.put(i, label);
    			}
    		}
        } catch (IOException e) {
        	System.err.println(e.getMessage());
        }
	}

	@Test
	public void MSE() {
		Problem problem = new Problem(X, Y, theta);
		MSEOptimizer lnfunc = new MSEOptimizer(problem);
		BFGS.train(lnfunc, 1000, 0.00001).print();
		LogisticRegressionOptimizer lrFunc = new LogisticRegressionOptimizer(problem);
		BFGS.train(lrFunc, 1000, 0.00001).print();
	}
}
