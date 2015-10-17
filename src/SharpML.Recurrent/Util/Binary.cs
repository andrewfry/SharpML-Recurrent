using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;

namespace SharpML.Recurrent.Util
{
    public static class Binary
    {
        public static T ReadFromBinary<T>(string filePath)
        {
            if (File.Exists(filePath))
            {
                using (var fs = new FileStream(filePath, FileMode.Open))
                {
                    BinaryFormatter formatter = new BinaryFormatter();
                    formatter.AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple;
                    if (fs.Length == 0)
                        return default(T);

                    return ((T)formatter.Deserialize(fs));
                }
            }

            throw new FileNotFoundException(filePath);
        }

        public static void WriteToBinary<T>(object dataToWrite, string filePath)
        {
            using (Stream stream = File.Open(filePath, FileMode.Create))
            {
                BinaryFormatter bformatter = new BinaryFormatter();
                bformatter.AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple;

                var filesToWrite = ((T)dataToWrite);

                bformatter.Serialize(stream, filesToWrite);
            }
        }

    }
}
