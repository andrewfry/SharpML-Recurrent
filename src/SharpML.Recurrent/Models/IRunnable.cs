using System;

namespace SharpML.Recurrent.Models
{
    public interface IRunnable
    {
        Action Run { get; set; }
    }
}
