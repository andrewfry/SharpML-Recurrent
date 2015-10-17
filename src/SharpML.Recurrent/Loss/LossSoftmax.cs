using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent;
using SharpML.Recurrent.DataStructs;
using SharpML.Recurrent.Models;
using SharpML.Recurrent.Networks;

namespace SharpML.Recurrent.Loss
{
    public class LossSoftmax : ILoss {

	public void Backward(Matrix logprobs, Matrix targetOutput)  {
		int targetIndex = GetTargetIndex(targetOutput);
		Matrix probs = GetSoftmaxProbs(logprobs, 1.0);
		for (int i = 0; i < probs.W.Length; i++) {
			logprobs.Dw[i] = probs.W[i];
		}
		logprobs.Dw[targetIndex] -= 1;
	}

	public double Measure(Matrix logprobs, Matrix targetOutput)  {
		int targetIndex = GetTargetIndex(targetOutput);
		Matrix probs = GetSoftmaxProbs(logprobs, 1.0);
		double cost = -Math.Log(probs.W[targetIndex]);
		return cost;
	}

	public static double CalculateMedianPerplexity(ILayer layer, List<DataSequence> sequences) {
		double temperature = 1.0;
        List<Double> ppls = new List<Double>();
		foreach (DataSequence seq in sequences) {
			double n = 0;
			double neglog2Ppl = 0;
			
			Graph g = new Graph(false);
			layer.ResetState();
			foreach (DataStep step in seq.Steps) {
				Matrix logprobs = layer.Activate(step.Input, g);
				Matrix probs = GetSoftmaxProbs(logprobs, temperature);
				int targetIndex = GetTargetIndex(step.TargetOutput);
				double probOfCorrect = probs.W[targetIndex];
				double log2Prob = Math.Log(probOfCorrect)/Math.Log(2); //change-of-base
				neglog2Ppl += -log2Prob;
				n += 1;
			}
			
			n -= 1; //don't count first symbol of sentence
			double ppl = Math.Pow(2, (neglog2Ppl/(n-1)));
			ppls.Add(ppl);
		}
        return Util.Util.Median(ppls);
	}
	
	public static Matrix GetSoftmaxProbs(Matrix logprobs, double temperature) {	
		Matrix probs = new Matrix(logprobs.W.Length);
		if (temperature != 1.0) {
			for (int i = 0; i < logprobs.W.Length; i++) {
				logprobs.W[i] /= temperature;
			}
		}
		double maxval = Double.NegativeInfinity;
		for (int i = 0; i < logprobs.W.Length; i++) {
			if (logprobs.W[i] > maxval) {
				maxval = logprobs.W[i];
			}
		}
		double sum = 0;
		for (int i = 0; i < logprobs.W.Length; i++) {
			probs.W[i] = Math.Exp(logprobs.W[i] - maxval); //all inputs to exp() are non-positive
			sum += probs.W[i];
		}
		for (int i = 0; i < probs.W.Length; i++) {
			probs.W[i] /= sum;
		}
		return probs;
	}

	private static int GetTargetIndex(Matrix targetOutput)  {
		for (int i = 0; i < targetOutput.W.Length; i++) {
			if (targetOutput.W[i] == 1.0) {
				return i;
			}
		}
		throw new Exception("no target index selected");
	}
}
}
