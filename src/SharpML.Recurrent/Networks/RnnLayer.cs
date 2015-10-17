using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
     [Serializable]
    public class RnnLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
        private int _inputDimension;
        private readonly int _outputDimension;

        private readonly Matrix _w;
         private readonly Matrix _b;

         private Matrix _context;

        private readonly INonlinearity _f;

        public RnnLayer(int inputDimension, int outputDimension, INonlinearity hiddenUnit, double initParamsStdDev,
            Random rng)
        {
            this._inputDimension = inputDimension;
            this._outputDimension = outputDimension;
            this._f = hiddenUnit;
            _w = Matrix.Random(outputDimension, inputDimension + outputDimension, initParamsStdDev, rng);
            _b = new Matrix(outputDimension);
        }

        public Matrix Activate(Matrix input, Graph g)
        {
            Matrix concat = g.ConcatVectors(input, _context);
            Matrix sum = g.Mul(_w, concat); sum = g.Add(sum, _b);
            Matrix output = g.Nonlin(_f, sum);

            //rollover activations for next iteration
            _context = output;

            return output;
        }


        public void ResetState()
        {
            _context = new Matrix(_outputDimension);
        }


        public List<Matrix> GetParameters()
        {
            List<Matrix> result = new List<Matrix>();
            result.Add(_w);
            result.Add(_b);
            return result;
        }
    }
}
