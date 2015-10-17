using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Loss
{
    public class LossArgMax : ILoss
    {

        public void Backward(Matrix actualOutput, Matrix targetOutput)
        {
            throw new Exception("not implemented");

        }

        public double Measure(Matrix actualOutput, Matrix targetOutput)
        {
            if (actualOutput.W.Length != targetOutput.W.Length)
            {
                throw new Exception("mismatch");
            }
            double maxActual = Double.PositiveInfinity;
            double maxTarget = Double.NegativeInfinity;
            int indxMaxActual = -1;
            int indxMaxTarget = -1;
            for (int i = 0; i < actualOutput.W.Length; i++)
            {
                if (actualOutput.W[i] > maxActual)
                {
                    maxActual = actualOutput.W[i];
                    indxMaxActual = i;
                }
                if (targetOutput.W[i] > maxTarget)
                {
                    maxTarget = targetOutput.W[i];
                    indxMaxTarget = i;
                }
            }
            if (indxMaxActual == indxMaxTarget)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }
    }
}
