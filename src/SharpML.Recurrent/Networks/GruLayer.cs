using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
     [Serializable]
    public class GruLayer : ILayer {

	private static  long _serialVersionUid = 1L;
	int _inputDimension;
         readonly int _outputDimension;

         readonly Matrix _hmix;
         readonly Matrix _hHmix;
         readonly Matrix _bmix;
         readonly Matrix _hnew;
         readonly Matrix _hHnew;
         readonly Matrix _bnew;
         readonly Matrix _hreset;
         readonly Matrix _hHreset;
         readonly Matrix _breset;

         Matrix _context;

         readonly INonlinearity _fMix = new SigmoidUnit();
         readonly INonlinearity _fReset = new SigmoidUnit();
         readonly INonlinearity _fNew = new TanhUnit();
	
	public GruLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng) {
		this._inputDimension = inputDimension;
		this._outputDimension = outputDimension;
		_hmix = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
		_hHmix = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
		_bmix = new Matrix(outputDimension);
		_hnew = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
		_hHnew = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
		_bnew = new Matrix(outputDimension);
		_hreset = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
		_hHreset = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
		_breset= new Matrix(outputDimension);
	}
	
	public Matrix Activate(Matrix input, Graph g)  {
		
		Matrix sum0 = g.Mul(_hmix, input);
		Matrix sum1 = g.Mul(_hHmix, _context);
		Matrix actMix = g.Nonlin(_fMix, g.Add(g.Add(sum0, sum1), _bmix));

		Matrix sum2 = g.Mul(_hreset, input);
		Matrix sum3 = g.Mul(_hHreset, _context);
		Matrix actReset = g.Nonlin(_fReset, g.Add(g.Add(sum2, sum3), _breset));
		
		Matrix sum4 = g.Mul(_hnew, input);
		Matrix gatedContext = g.Elmul(actReset, _context);
		Matrix sum5 = g.Mul(_hHnew, gatedContext);
		Matrix actNewPlusGatedContext = g.Nonlin(_fNew, g.Add(g.Add(sum4, sum5), _bnew));
		
		Matrix memvals = g.Elmul(actMix, _context);
		Matrix newvals = g.Elmul(g.OneMinus(actMix), actNewPlusGatedContext);
		Matrix output = g.Add(memvals, newvals);
		
		//rollover activations for next iteration
		_context = output;
		
		return output;
	}

	public void ResetState() {
		_context = new Matrix(_outputDimension);
	}

	public List<Matrix> GetParameters() {
		List<Matrix> result = new List<Matrix>();
		result.Add(_hmix);
        result.Add(_hHmix);
        result.Add(_bmix);
        result.Add(_hnew);
        result.Add(_hHnew);
        result.Add(_bnew);
        result.Add(_hreset);
        result.Add(_hHreset);
        result.Add(_breset);
		return result;
	}

}
}
