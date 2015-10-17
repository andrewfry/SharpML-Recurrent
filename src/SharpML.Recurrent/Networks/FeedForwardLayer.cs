using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    [Serializable]
    public class FeedForwardLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
        readonly Matrix _w;
        readonly Matrix _b;
        readonly INonlinearity _f;

        public FeedForwardLayer(int inputDimension, int outputDimension, INonlinearity f, double initParamsStdDev, Random rng)
        {
            _w = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
            _b = new Matrix(outputDimension);
            this._f = f;
        }

        public Matrix Activate(Matrix input, Graph g)
        {
            Matrix sum = g.Add(g.Mul(_w, input), _b);
            Matrix returnObj = g.Nonlin(_f, sum);
            return returnObj;
        }

        public void ResetState()
        {

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
