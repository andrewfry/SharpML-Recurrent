using System;

namespace SharpML.Recurrent.Activations
{
    [Serializable]
    public class SigmoidUnit : INonlinearity
    {
        private static long _serialVersionUid = 1L;
        private readonly long _id;

        public long Id
        {
            get { return _id; }
        }

        public SigmoidUnit()
        {
            _id = _serialVersionUid + 1;
        }

        public double Forward(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public double Backward(double x)
        {
            double act = Forward(x);
            return act * (1 - act);
        }
    }
}
