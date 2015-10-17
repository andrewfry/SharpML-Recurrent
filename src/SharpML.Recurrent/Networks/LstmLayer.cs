using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    [Serializable]
    public class LstmLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
        int _inputDimension;
        readonly int _outputDimension;

        readonly Matrix _wix;
        readonly Matrix _wih;
        readonly Matrix _inputBias;
        readonly Matrix _wfx;
        readonly Matrix _wfh;
        readonly Matrix _forgetBias;
        readonly Matrix _wox;
        readonly Matrix _woh;
        readonly Matrix _outputBias;
        readonly Matrix _wcx;
        readonly Matrix _wch;
        readonly Matrix _cellWriteBias;

        Matrix _hiddenContext;
        Matrix _cellContext;

        readonly INonlinearity _inputGateActivation = new SigmoidUnit();
        readonly INonlinearity _forgetGateActivation = new SigmoidUnit();
        readonly INonlinearity _outputGateActivation = new SigmoidUnit();
        readonly INonlinearity _cellInputActivation = new TanhUnit();
        readonly INonlinearity _cellOutputActivation = new TanhUnit();

        public LstmLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng)
        {
            this._inputDimension = inputDimension;
            this._outputDimension = outputDimension;
            _wix = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wih = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _inputBias = new Matrix(outputDimension);
            _wfx = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wfh = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            //set forget bias to 1.0, as described here: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
            _forgetBias = Matrix.Ones(outputDimension, 1);
            _wox = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _woh = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _outputBias = new Matrix(outputDimension);
            _wcx = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _wch = Matrix.Random(outputDimension, outputDimension, initParamsStdDev, rng);
            _cellWriteBias = new Matrix(outputDimension);
        }

        public Matrix Activate(Matrix input, Graph g)
        {

            //input gate
            Matrix sum0 = g.Mul(_wix, input);
            Matrix sum1 = g.Mul(_wih, _hiddenContext);
            Matrix inputGate = g.Nonlin(_inputGateActivation, g.Add(g.Add(sum0, sum1), _inputBias));

            //forget gate
            Matrix sum2 = g.Mul(_wfx, input);
            Matrix sum3 = g.Mul(_wfh, _hiddenContext);
            Matrix forgetGate = g.Nonlin(_forgetGateActivation, g.Add(g.Add(sum2, sum3), _forgetBias));

            //output gate
            Matrix sum4 = g.Mul(_wox, input);
            Matrix sum5 = g.Mul(_woh, _hiddenContext);
            Matrix outputGate = g.Nonlin(_outputGateActivation, g.Add(g.Add(sum4, sum5), _outputBias));

            //write operation on cells
            Matrix sum6 = g.Mul(_wcx, input);
            Matrix sum7 = g.Mul(_wch, _hiddenContext);
            Matrix cellInput = g.Nonlin(_cellInputActivation, g.Add(g.Add(sum6, sum7), _cellWriteBias));

            //compute new cell activation
            Matrix retainCell = g.Elmul(forgetGate, _cellContext);
            Matrix writeCell = g.Elmul(inputGate, cellInput);
            Matrix cellAct = g.Add(retainCell, writeCell);

            //compute hidden state as gated, saturated cell activations
            Matrix output = g.Elmul(outputGate, g.Nonlin(_cellOutputActivation, cellAct));

            //rollover activations for next iteration
            _hiddenContext = output;
            _cellContext = cellAct;

            return output;
        }

        public void ResetState()
        {
            _hiddenContext = new Matrix(_outputDimension);
            _cellContext = new Matrix(_outputDimension);
        }

        public List<Matrix> GetParameters()
        {
            List<Matrix> result = new List<Matrix>();
            result.Add(_wix);
            result.Add(_wih);
            result.Add(_inputBias);
            result.Add(_wfx);
            result.Add(_wfh);
            result.Add(_forgetBias);
            result.Add(_wox);
            result.Add(_woh);
            result.Add(_outputBias);
            result.Add(_wcx);
            result.Add(_wch);
            result.Add(_cellWriteBias);
            return result;
        }
    }
}
