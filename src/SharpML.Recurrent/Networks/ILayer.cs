using System.Collections.Generic;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Networks
{
    public interface ILayer 
    {
        Matrix Activate(Matrix input, Graph g);
        void ResetState();
        List<Matrix> GetParameters();
    }
}
