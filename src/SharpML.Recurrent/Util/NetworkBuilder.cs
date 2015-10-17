using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Networks;

namespace SharpML.Recurrent.Util
{
    public static class NetworkBuilder
    {

        public static NeuralNetwork MakeLstm(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new LstmLayer(inputDimension, hiddenDimension, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new LstmLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static NeuralNetwork MakeLstmWithInputBottleneck(int inputDimension, int bottleneckDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            layers.Add(new LinearLayer(inputDimension, bottleneckDimension, initParamsStdDev, rng));
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new LstmLayer(bottleneckDimension, hiddenDimension, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new LstmLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static NeuralNetwork MakeFeedForward(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity hiddenUnit, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new FeedForwardLayer(inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new FeedForwardLayer(hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static NeuralNetwork MakeGru(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new GruLayer(inputDimension, hiddenDimension, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new GruLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }

        public static NeuralNetwork MakeRnn(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, INonlinearity hiddenUnit, INonlinearity decoderUnit, double initParamsStdDev, Random rng)
        {
            List<ILayer> layers = new List<ILayer>();
            for (int h = 0; h < hiddenLayers; h++)
            {
                if (h == 0)
                {
                    layers.Add(new RnnLayer(inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
                }
                else
                {
                    layers.Add(new RnnLayer(hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
                }
            }
            layers.Add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
            return new NeuralNetwork(layers);
        }
    }
}
