using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.CPlusPlus;
using Point = OpenCvSharp.Point;
using Mat = OpenCvSharp.Mat;
using OpenCvSharp.Text;
using System.Diagnostics.Tracing;
using grabCut_JTLU;
using Accord.Math;


namespace grabCut
{
    class grabCut
    {
        public Rect rect;
        public Mat leftW, upleftW, upW, uprightW;

        public Mat ImageRead(string path)
        {
            try
            {
                string baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                string relativePath = path;
                Console.WriteLine(Path.Combine(baseDir, relativePath));
                string fullPath = Path.GetFullPath(Path.Combine(baseDir, relativePath));
                Mat imgLoad = new Mat(fullPath, ImreadModes.Color);

                if (imgLoad.Empty())
                {
                    Console.WriteLine("Image not found or could not be loaded.");
                    Console.ReadKey();
                    return null;
                }
                return imgLoad;
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred: " + ex.Message);
            }
            return null;
        }
        
        private void InitializeWithRect(Mat img, Mat mask, Rect rect)
        {
            // Background: 0, PossibleBackground: 2, Foreground: 1, PossibleForeground: 3

            // initialize the mask with all possible background
            mask.SetTo(new Scalar((int)GrabCutClasses.PR_BGD));
            Rect foregroundRegion = new Rect(rect.X, rect.Y, rect.Width, rect.Height);

            // inside the rectangle is possible foreground
            Mat foregroundMaskROI = new Mat(mask, foregroundRegion);
            foregroundMaskROI.SetTo(new Scalar((int)GrabCutClasses.PR_FGD));
        }

        private void VisMask(Mat mask)
        {
            Mat visualization = new Mat(mask.Size(), MatType.CV_8UC3);
            int cntFG = 0, cntBG = 0;

            for (int y = 0; y < mask.Rows; y++)
            {
                for (int x = 0; x < mask.Cols; x++)
                {
                    byte maskValue = mask.At<byte>(y, x);
                    Vec3b color = new Vec3b();
                    switch (maskValue)
                    {
                        case (byte)GrabCutClasses.PR_FGD: // 可能的前景
                            color = new Vec3b(255, 255, 255); // 白色
                            cntFG++;
                            break;
                        case (byte)GrabCutClasses.PR_BGD:
                            color = new Vec3b(64, 64, 64);
                            cntBG++;
                            break;
                        default:
                            color = new Vec3b(0, 0, 0);
                            break;
                    }

                    visualization.Set(y, x, color);
                }
            }
            Console.WriteLine("cnt FG BG "+ cntFG + " , "+ cntBG);
            Cv2.ImShow("Mask_Visualization", visualization);
        }

        public void JTLU_GrabCut(Mat img, Mat mask, Rect rect, Mat bgdModel, Mat fgdModel, int iterCount, GrabCutModes mode)
        {
            if (mode == GrabCutModes.InitWithRect)
            {
                InitializeWithRect(img, mask, rect);
            }
            else
            {
                throw new ArgumentException("Unsupported GrabCut mode.");
            }
            GMM gmmModel = new GMM();
            gmmModel.Initialize(img, mask, rect);

            for (int i = 0; i < iterCount; i++)
            {
                // 1. Assign GMM components to pixels
                gmmModel.AssignGMMComponents(img, mask);
                
                // 2.Learn and Update GMM parameters
                gmmModel.Fit(img, mask);

                // 3.Construct graph
                Graph graph = new Graph(2,0,1);
                graph.ConstructGraph(img, mask, gmmModel, leftW, upleftW, upW, uprightW);
                Console.WriteLine("SourceID "+ graph.Source + " sink "+ graph.Sink);
                // 4. Estimate segmentation, and mask updating
                graph.EstimateSegmentation(graph, mask);
            }
            VisMask(mask);
        }
        public static double CalcBeta(Mat img)
        {
            double beta = 0;
            int rows = img.Rows;
            int cols = img.Cols;
            int channels = img.Channels();

            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    Vec3b color = img.At<Vec3b>(y, x);
                    if (x > 0) // left
                    {
                        Vec3b leftColor = img.At<Vec3b>(y, x - 1);
                        beta += CalcColorDiff(color, leftColor);
                    }
                    if (y > 0 && x > 0) // upleft
                    {
                        Vec3b upLeftColor = img.At<Vec3b>(y - 1, x - 1);
                        beta += CalcColorDiff(color, upLeftColor);
                    }
                    if (y > 0) // up
                    {
                        Vec3b upColor = img.At<Vec3b>(y - 1, x);
                        beta += CalcColorDiff(color, upColor);
                    }
                    if (y > 0 && x < cols - 1) // upright
                    {
                        Vec3b upRightColor = img.At<Vec3b>(y - 1, x + 1);
                        beta += CalcColorDiff(color, upRightColor);
                    }
                }
            }

            int normFactor = (4 * cols * rows) - (3 * cols) - (3 * rows) + 2;
            beta = (beta <= double.Epsilon) ? 0 : 1.0 / (2.0 * beta / normFactor);

            return beta;
        }

        private static double CalcColorDiff(Vec3b color1, Vec3b color2)
        {
            double diff0 = color1.Item0 - color2.Item0;
            double diff1 = color1.Item1 - color2.Item1;
            double diff2 = color1.Item2 - color2.Item2;
            return (diff0 * diff0) + (diff1 * diff1) + (diff2 * diff2);
        }


        static void calcNWeights(Mat img, out Mat leftW, out Mat upleftW, out Mat upW, out Mat uprightW, double beta, double gamma)
        {
            double gammaDivSqrt2 = gamma / Math.Sqrt(2.0);
            leftW = new Mat(img.Rows, img.Cols, MatType.CV_64FC1);
            upleftW = new Mat(img.Rows, img.Cols, MatType.CV_64FC1);
            upW = new Mat(img.Rows, img.Cols, MatType.CV_64FC1);
            uprightW = new Mat(img.Rows, img.Cols, MatType.CV_64FC1);

            for (int y = 0; y < img.Rows; y++)
            {
                for (int x = 0; x < img.Cols; x++)
                {
                    Vec3b color = img.At<Vec3b>(y, x);
                    if (x - 1 >= 0) // left
                    {
                        double diff = CalcColorDiff(color, img.At<Vec3b>(y, x - 1));
                        leftW.Set(y, x, gamma * Math.Exp(-beta * diff));
                    }
                    else
                    {
                        leftW.Set(y, x, 0);
                    }
                    if (x - 1 >= 0 && y - 1 >= 0) // upleft
                    {
                        double diff = CalcColorDiff(color, img.At<Vec3b>(y - 1, x - 1));
                        upleftW.Set(y, x, gammaDivSqrt2 * Math.Exp(-beta * diff));
                    }
                    else
                    {
                        upleftW.Set(y, x, 0);
                    }
                    if (y - 1 >= 0) // up
                    {
                        double diff = CalcColorDiff(color, img.At<Vec3b>(y - 1, x));
                        upW.Set(y, x, gamma * Math.Exp(-beta * diff));
                    }
                    else
                    {
                        upW.Set(y, x, 0);
                    }
                    if (x + 1 < img.Cols && y - 1 >= 0) // upright
                    {
                        double diff = CalcColorDiff(color, img.At<Vec3b>(y - 1, x + 1));
                        uprightW.Set(y, x, gammaDivSqrt2 * Math.Exp(-beta * diff));
                    }
                    else
                    {
                        uprightW.Set(y, x, 0);
                    }
                }
            }
        }

        public void GrabCut(Mat img)
        {
            Mat mask = new Mat(img.Size(), MatType.CV_8UC1, new Scalar(0));
            Mat bgdModel = new Mat();
            Mat fgdModel = new Mat();


            Mat imgBbox = img.Clone();

            Cv2.Rectangle(imgBbox, rect, new Scalar(0, 0, 255), 2);
            Cv2.ImShow("Rect Vis", imgBbox);
            Console.WriteLine("Press AnyKey on the Image to Begin the GrabCut");
            Cv2.WaitKey(0);

            double gamma = 50.0;
            double beta = CalcBeta(img);
            // beta *= 10f;
            Console.WriteLine("beta params of the img  "+ beta);
            calcNWeights(img, out leftW, out upleftW, out upW, out uprightW, beta, gamma);

            // Cv2.GrabCut(img, mask, rect, bgdModel, fgdModel, 5, GrabCutModes.InitWithRect);
            JTLU_GrabCut(img, mask, rect, bgdModel, fgdModel, 2, GrabCutModes.InitWithRect);
            for (int y = 0; y < mask.Rows; y++)
            {
                for (int x = 0; x < mask.Cols; x++)
                {

                    if (x < rect.X - 40 || x > rect.X + rect.Width + 40 || y < rect.Y - 40 || y > rect.Y + rect.Height + 40)
                    {
                        if (mask.At<byte>(y, x) == new Scalar(3))
                        {
                            mask.Set<byte>(y, x, (byte)GrabCutClasses.PR_BGD);
                        }
                    }
                }
            }


            Mat result = new Mat();
            Cv2.Compare(mask, new Scalar(3), result, CmpType.EQ);
            Mat foreground = new Mat(img.Size(), MatType.CV_8UC3, new Scalar(255, 255, 255));
            img.CopyTo(foreground, result);
            Cv2.ImShow("Foreground", foreground);
            Cv2.WaitKey(0);
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            grabCut gbCut = new grabCut();
            string PATH = @"..\\..\\..\\imgs\\nvidia.jpg";
            Mat imgLoad = gbCut.ImageRead(PATH);
            // gbCut.rect = new Rect(230, 120, 360, 280);
            gbCut.rect = new Rect(190, 80, 260, 180);
            gbCut.GrabCut(imgLoad);
        }
    }
    
}

