using System;
using System.Collections.Generic;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    [Serializable]
    public class NeuralNetwork : INetwork
    {

        private static long _serialVersionUid = 1L;
        readonly List<ILayer> _layers;

        public NeuralNetwork(List<ILayer> layers)
        {
            this._layers = layers;
        }

        public Matrix Activate(Matrix input, Graph g)
        {
            Matrix prev = input;
            foreach (ILayer layer in _layers)
            {
                prev = layer.Activate(prev, g);
            }
            return prev;
        }

        public void ResetState()
        {
            foreach (ILayer layer in _layers)
            {
                layer.ResetState();
            }
        }

        public List<Matrix> GetParameters()
        {
            List<Matrix> result = new List<Matrix>();
            foreach (ILayer layer in _layers)
            {
                result.AddRange(layer.GetParameters());
            }
            return result;
        }
    }
}
