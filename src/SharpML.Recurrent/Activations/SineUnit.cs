using System;

namespace SharpML.Recurrent.Activations
{
    [Serializable]
    public class SineUnit : INonlinearity
    {
        private static long _serialVersionUid = 1L;
        private readonly long _id;

        public long Id
        {
            get { return _id; }
        }

        public SineUnit()
        {
            _id = _serialVersionUid + 1;
        }

        public double Forward(double x)
        {
            return Math.Sin(x);
        }

        public double Backward(double x)
        {
            return Math.Cos(x);
        }
    }
}
