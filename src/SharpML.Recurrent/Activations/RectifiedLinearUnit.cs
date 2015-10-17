using System;

namespace SharpML.Recurrent.Activations
{
    [Serializable]
    public class RectifiedLinearUnit : INonlinearity
    {

        private static long _serialVersionUid = 1L;
        private readonly double _slope;
        private readonly long _id;

        public long Id
        {
            get { return _id; }
        }

        public RectifiedLinearUnit()
        {
            _id = _serialVersionUid + 1;
            this._slope = 0;
        }

        public RectifiedLinearUnit(double slope)
        {
            _id = _serialVersionUid + 1;
            this._slope = slope;
        }

        public double Forward(double x)
        {
            if (x >= 0)
            {
                return x;
            }
            else
            {
                return x * _slope;
            }
        }

        public double Backward(double x)
        {
            if (x >= 0)
            {
                return 1.0;
            }
            else
            {
                return _slope;
            }
        }
    }
}
