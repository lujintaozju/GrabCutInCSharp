using Accord.MachineLearning;
using Accord.Math;
using Accord.Statistics;
using Accord.Statistics.Distributions.Multivariate;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace grabCut_JTLU
{
    public class GMM
    {
        public GaussianMixtureModel gmmForeground;
        public GaussianMixtureModel gmmBackground;
        public GaussianClusterCollection clustersForeground;
        public GaussianClusterCollection clustersBackground;

        private const int componentsCount = 4;

        public GMM()
        {
            gmmForeground = new GaussianMixtureModel(componentsCount);
            gmmBackground = new GaussianMixtureModel(componentsCount);
        }

        private double[][] MatToJagged(Mat image, Mat mask, byte maskValue)
        {
            List<double[]> samples = new List<double[]>();
            for (int y = 0; y < image.Rows; y++)
            {
                for (int x = 0; x < image.Cols; x++)
                {
                    if (mask.At<byte>(y, x) == maskValue)
                    {
                        Vec3b color = image.At<Vec3b>(y, x);
                        samples.Add(new double[] { color.Item0, color.Item1, color.Item2 });
                    }
                }
            }
            return samples.ToArray();
        }

        public void Initialize(Mat image, Mat mask, Rect rect)
        {
            // Convert image to data points for the GMM
            double[][] samplesForeground = MatToJagged(image, mask, (byte)GrabCutClasses.PR_FGD);
            double[][] samplesBackground = MatToJagged(image, mask, (byte)GrabCutClasses.PR_BGD);
            Console.WriteLine("Init samplesForeground Num " + samplesForeground.Length);
            Console.WriteLine("Init samplesBackground Num " + samplesBackground.Length);
            // Learn the GMM parameters for the foreground
            if (samplesForeground.Length > 0)
            {
                clustersForeground = gmmForeground.Learn(samplesForeground);
            }

            // Learn the GMM parameters for the background
            if (samplesBackground.Length > 0)
            {
                clustersBackground = gmmBackground.Learn(samplesBackground);
            }
        }

        public void Fit(Mat image, Mat mask)
        {
            // Convert image to data points for the GMM
            double[][] samplesForeground = MatToJagged(image, mask, (byte)GrabCutClasses.PR_FGD);
            double[][] samplesBackground = MatToJagged(image, mask, (byte)GrabCutClasses.PR_BGD);

            // Re-estimate GMM parameters for the foreground
            if (samplesForeground.Length > 0)
            {
                gmmForeground = new GaussianMixtureModel(componentsCount);
                clustersForeground = gmmForeground.Learn(samplesForeground);
            }

            // Re-estimate GMM parameters for the background
            if (samplesBackground.Length > 0)
            {
                gmmBackground = new GaussianMixtureModel(componentsCount);
                clustersBackground = gmmBackground.Learn(samplesBackground);
            }
        }


        public double ComputePDF(GaussianMixtureModel gmm, double[] dataPoint, bool print = false)
        {
            double probability = 0;
            for (int i = 0; i < gmm.Gaussians.Count; i++)
            {
                double likelihood = -0.01 * gmm.Gaussians[i].LogLikelihood(dataPoint);
                probability += likelihood;
            }
            if (print)
            {
                Console.WriteLine("GMM Gaussian " + dataPoint[0] + "," + dataPoint[1] + "," + dataPoint[2] + "  probs "+ probability);
            }
            return probability;
        }

        public void AssignGMMComponents(Mat img, Mat mask)
        {
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    Vec3b color = img.At<Vec3b>(y, x);
                    double[] sample = new double[]
                    {
                        color.Item0,
                        color.Item1,
                        color.Item2
                    };

                    bool print = false;
                    if (x == img.Width / 2 && y == img.Height / 2)
                    {
                        print = true;
                    }

                    //Need compute the probs for FG/BG
                    double probsForeground = ComputePDF(gmmForeground, sample, print);
                    double probsBackground = ComputePDF(gmmBackground, sample, print);

                    // Assign to the mask the most probable component
                    if (probsForeground > probsBackground)
                    {
                        if (y % 50 == 0 && x % 50 == 0)
                        {
                            Console.WriteLine("set as possible FGD by PDF  "+ x+","+y);
                        }
                        mask.Set<byte>(y, x, (byte)GrabCutClasses.PR_FGD); // Probable foreground
                    }
                    else
                    {
                        mask.Set<byte>(y, x, (byte)GrabCutClasses.PR_BGD); // Probable background
                    }
                }
            }
        }
    }
}