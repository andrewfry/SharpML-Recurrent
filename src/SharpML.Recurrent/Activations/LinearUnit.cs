using System;

namespace SharpML.Recurrent.Activations
{
    [Serializable]
    public class LinearUnit : INonlinearity
    {
        private static long _serialVersionUid = 1L;
        private readonly long _id;

        public long  Id {
            get { return _id; }
        }

        public LinearUnit()
        {
            _id = _serialVersionUid + 1;
        }

        public double Forward(double x)
        {
            return x;
        }

        public double Backward(double x)
        {
            return 1.0;
        }
    }
}
