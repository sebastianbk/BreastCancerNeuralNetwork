using System;
using System.Collections.Generic;
using System.IO;

namespace BreastCancerNeuralNetwork
{
    class Program
    {
        #region Input Variables
        // Path to source data file
        private static readonly string sourceFile = Path.Combine(Environment.CurrentDirectory, "breast-cancer-wisconsin.csv");

        // Number of input neurons, hidden neurons and output neurons
        private static readonly int[] inputColumns = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        private static readonly int numInput = inputColumns.Length;
        private const int numHidden = 7;
        private const int numOutput = 2;

        // Parameters for NN training
        private const int maxEpochs = 2000;
        private const double learnRate = 0.05;
        private const double momentum = 0.01;
        private const double weightDecay = 0.0001;
        #endregion

        static void Main(string[] args)
        {
            Console.WriteLine("Neural Network Demo using .NET by Sebastian Brandes");
            Console.WriteLine("Data Set: Breast Cancer Wisconsin (Diagnostic), November 1995");
            // Source: http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
            Console.WriteLine();

            #region Data Generation
            Console.WriteLine("Loading source file and generating data sets...");
            var rows = File.ReadAllLines(sourceFile);
            var data = new List<double[]>();

            foreach (var row in rows)
            {
                var values = row.Split(',');
                var observation = new double[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    double.TryParse(values[i], out observation[i]);
                }
                data.Add(observation);
            }

            List<double[]> trainData;
            List<double[]> testData;
            Helpers.GenerateDataSets(data, out trainData, out testData, 0.8);

            Console.WriteLine("Done!");
            Console.WriteLine();
            #endregion

            #region Normalization
            Console.WriteLine("Normalizing data...");
            List<double[]> normalizedTrainData = Helpers.NormalizeData(trainData, inputColumns);
            List<double[]> normalizedTestData = Helpers.NormalizeData(testData, inputColumns);

            Console.WriteLine("Done!");
            Console.WriteLine();
            #endregion

            #region Initializing the Neural Network
            Console.WriteLine("Creating a new {0}-input, {1}-hidden, {2}-output neural network...", numInput, numHidden, numOutput);
            var nn = new NeuralNetwork(numInput, numHidden, numOutput);

            Console.WriteLine("Initializing weights and bias to small random values...");
            nn.InitializeWeights();

            Console.WriteLine("Done!");
            Console.WriteLine();
            #endregion

            #region Training
            Console.WriteLine("Beginning training using incremental back-propagation...");
            nn.Train(normalizedTrainData.ToArray(), maxEpochs, learnRate, momentum, weightDecay);

            Console.WriteLine("Done!");
            Console.WriteLine();
            #endregion

            #region Results
            double[] weights = nn.GetWeights();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Final neural network weights and bias values:");
            Console.ResetColor();
            Helpers.ShowVector(weights, 10, 3, true);
            Console.WriteLine();

            double trainAcc = nn.Accuracy(normalizedTrainData.ToArray());
            Console.WriteLine("Accuracy on training data = " + trainAcc.ToString("F4"));
            double testAcc = nn.Accuracy(normalizedTestData.ToArray());
            Console.WriteLine("Accuracy on test data = " + testAcc.ToString("F4"));
            Console.WriteLine();

            //Console.ForegroundColor = ConsoleColor.Green;
            //Console.WriteLine("Raw results:");
            //Console.ResetColor();
            //Console.WriteLine(nn.ToString());
            #endregion

            Console.ReadKey();
        }
    }
}
