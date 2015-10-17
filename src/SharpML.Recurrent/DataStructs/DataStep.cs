using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.DataStructs
{
    public class DataStep {

	public Matrix Input = null;
	public Matrix TargetOutput = null;
	
	public DataStep() {
		
	}
	
	public DataStep(double[] input, double[] targetOutput) {
		this.Input = new Matrix(input);
		if (targetOutput != null) {
			this.TargetOutput = new Matrix(targetOutput);
		}
	}
	
	public override string ToString() {
		String result = "";
		for (int i = 0; i < Input.W.Length; i++) {
            result += String.Format("{0:N5}", Input.W[i]) + "\t";
		}
		result += "\t->\t";
		if (TargetOutput != null) {
            for (int i = 0; i < TargetOutput.W.Length; i++)
            {
                result += String.Format("{0:N5}", TargetOutput.W[i]) + "\t";
			}
		}
		else {
			result += "___\t";
		}
		return result;
	}
}
}
