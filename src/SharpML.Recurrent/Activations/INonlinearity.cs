namespace SharpML.Recurrent.Activations
{
    public interface INonlinearity  {
	double Forward(double x);
	double Backward(double x);
}
}
