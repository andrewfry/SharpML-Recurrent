using System;
using System.Collections.Generic;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
     [Serializable]
    public class LinearLayer : ILayer
    {

        private static long _serialVersionUid = 1L;
         readonly Matrix _w;
        //no biases

        public LinearLayer(int inputDimension, int outputDimension, double initParamsStdDev, Random rng)
        {
            _w = Matrix.Random(outputDimension, inputDimension, initParamsStdDev, rng);
        }

        public Matrix Activate(Matrix input, Graph g)
        {
            Matrix returnObj = g.Mul(_w, input);
            return returnObj;
        }

        public void ResetState()
        {

        }

        public List<Matrix> GetParameters()
        {
            List<Matrix> result = new List<Matrix>();
            result.Add(_w);
            return result;
        }
    }
}
