using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Loss
{
    public class LossMultiDimensionalBinary : ILoss
    {

        public void Backward(Matrix actualOutput, Matrix targetOutput)
        {
            throw new NotImplementedException("not implemented");
        }

        public double Measure(Matrix actualOutput, Matrix targetOutput)
        {
            if (actualOutput.W.Length != targetOutput.W.Length)
            {
                throw new Exception("mismatch");
            }

            for (int i = 0; i < targetOutput.W.Length; i++)
            {
                if (targetOutput.W[i] >= 0.5 && actualOutput.W[i] < 0.5)
                {
                    return 1;
                }
                if (targetOutput.W[i] < 0.5 && actualOutput.W[i] >= 0.5)
                {
                    return 1;
                }
            }
            return 0;
        }

    }
}
