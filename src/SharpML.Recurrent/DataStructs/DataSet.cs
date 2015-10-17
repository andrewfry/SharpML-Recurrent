using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Activations;
using SharpML.Recurrent.Loss;
using SharpML.Recurrent.Networks;

namespace SharpML.Recurrent.DataStructs
{
    public abstract class DataSet 
    {
        public int InputDimension { get; set; }
        public int OutputDimension { get; set; }
        public ILoss LossTraining { get; set; }
        public ILoss LossReporting { get; set; }
        public List<DataSequence> Training { get; set; }
        public List<DataSequence> Validation { get; set; }
        public List<DataSequence> Testing { get; set; }

        public virtual void DisplayReport(INetwork network, Random rng)
        {

        }

        public virtual INonlinearity GetModelOutputUnitToUse()
        {
            return null;
        }
}
}
