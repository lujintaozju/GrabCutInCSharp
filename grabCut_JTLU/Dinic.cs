using Accord.MachineLearning.Boosting;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
using System.Threading.Tasks;

namespace grabCut_JTLU
{

    public class Edge
    {
        public int To { get; set; }
        public int Capacity { get; set; }
        public int Flow { get; set; }
        public Edge Reverse { get; set; }

        public Edge(int to, int capacity)
        {
            To = to;
            Capacity = capacity;
            Flow = 0;
        }
    }

    public class Node
    {
        public List<Edge> Edges { get; set; } = new List<Edge>();
    }

    public class Graph
    {
        public Node[] Nodes { get; set; }
        public int Source { get; set; }
        public int Sink { get; set; }

        public Graph(int numberOfNodes, int source, int sink)
        {
            Nodes = new Node[numberOfNodes];
            for (int i = 0; i < Nodes.Length; i++)
            {
                Nodes[i] = new Node();
            }
            Source = source;
            Sink = sink;
        }

        public void AddEdge(int from, int to, int capacity)
        {
            Edge forwardEdge = new Edge(to, capacity);
            Edge reverseEdge = new Edge(from, 0); // Reverse edge has 0 initial capacity

            forwardEdge.Reverse = reverseEdge;
            reverseEdge.Reverse = forwardEdge;

            Nodes[from].Edges.Add(forwardEdge);
            Nodes[to].Edges.Add(reverseEdge);
        }

        private int ComputeNodeID(int x, int y, Mat img)
        {
            return (y* img.Cols + x);
        }

        public void ConstructGraph(Mat img, Mat mask, GMM gmm, Mat leftW, Mat upleftW, Mat upW, Mat uprightW)
        {
            // Additionally have the source and sink nodes
            int vertexCount = img.Rows * img.Cols + 2;
            int source = vertexCount - 2;
            int sink = vertexCount - 1;

            Nodes = new Node[vertexCount];
            for (int i = 0; i < Nodes.Length; i++)
            {
                Nodes[i] = new Node();
            }
            Source = source;
            Sink = sink;

            for (int y = 0; y < img.Rows; y++)
            {
                for (int x = 0; x < img.Cols; x++)
                {
                    int nodeIndex = ComputeNodeID(x, y, img);
                    Vec3b color = img.At<Vec3b>(y, x);
                    double[] sample = new double[] { color.Item0, color.Item1, color.Item2 };

                    double bgProb = gmm.ComputePDF(gmm.gmmBackground, sample);
                    double fgProb = gmm.ComputePDF(gmm.gmmForeground, sample);

                    int bgCapacity = (int)(bgProb);
                    int fgCapacity = (int)(fgProb);                 
                    AddEdge(source, nodeIndex, fgCapacity);
                    AddEdge(nodeIndex, sink, bgCapacity);
                    if (y%50 == 0 && x % 50 ==0 && fgCapacity > bgCapacity)
                        Console.WriteLine(" position " + y + "," + x + "  fgCap  " + fgCapacity + " bgCap  " + bgCapacity);
                    if (x > 0)
                    {
                        int w = (int) leftW.At<double>(y, x);
                        int neighborIndex = ComputeNodeID(x - 1, y, img);
                        AddEdge(nodeIndex, neighborIndex, w);
                        AddEdge(neighborIndex, nodeIndex, w);
                    }
                    if (x > 0 && y > 0)
                    {
                        int w = (int)upleftW.At<double>(y, x);
                        int neighborIndex = ComputeNodeID(x - 1, y - 1, img);
                        AddEdge(nodeIndex, neighborIndex, w);
                        AddEdge(neighborIndex, nodeIndex, w);
                    }
                    if (y > 0)
                    {
                        int w = (int)upW.At<double>(y, x);
                        int neighborIndex = ComputeNodeID(x, y - 1, img);
                        AddEdge(nodeIndex, neighborIndex, w);
                        AddEdge(neighborIndex, nodeIndex, w);
                    }
                    if (x < img.Cols - 1 && y>0)
                    {
                        int w = (int)uprightW.At<double>(y, x);
                        int neighborIndex = ComputeNodeID(x + 1, y - 1, img);
                        AddEdge(nodeIndex, neighborIndex, w);
                        AddEdge(neighborIndex, nodeIndex, w);
                    }
                }
            }
        }


        public int Dinic()
        {
            int flow = 0;
            int[] level = new int[Nodes.Length];
            while (BFS(level))
            {
                int[] start = new int[Nodes.Length];
                while (true)
                {
                    int f = DFS(Source, int.MaxValue, level, start);
                    if (f == 0) break;
                    flow += f;
                }
            }
            return flow;
        }

        public bool BFS(int[] level)
        {
            for (int i = 0; i < Nodes.Length; i++)
                level[i] = -1;

            Queue<int> queue = new Queue<int>();
            queue.Enqueue(Source);
            level[Source] = 0;

            while (queue.Count > 0)
            {
                int node = queue.Dequeue();
                foreach (var edge in Nodes[node].Edges)
                {
                    int cap = edge.Capacity - edge.Flow;
                    if (cap > 0 && level[edge.To] < 0)
                    {
                        level[edge.To] = level[node] + 1;
                        queue.Enqueue(edge.To);
                    }
                }
            }
            return level[Sink] >= 0;
        }

        public int DFS(int node, int flow, int[] level, int[] start)
        {
            if (node == Sink) return flow;

            for (; start[node] < Nodes[node].Edges.Count; start[node]++)
            {
                Edge edge = Nodes[node].Edges[start[node]];
                int cap = edge.Capacity - edge.Flow;
                if (cap > 0 && level[edge.To] == level[node] + 1)
                {
                    int f = DFS(edge.To, Math.Min(flow, cap), level, start);
                    if (f > 0)
                    {
                        edge.Flow += f;
                        edge.Reverse.Flow -= f;
                        return f;
                    }
                }
            }

            return 0;
        }

        public void EstimateSegmentation(Graph graph, Mat mask)
        {
            // Compute the MaxFlow
            int maxFlow = graph.Dinic();
            Console.WriteLine("MaxFlow "+ maxFlow);

            // BFS again and find the reachable nodes from the source
            HashSet<int> reachableFromSource = new HashSet<int>();
            Queue<int> queue = new Queue<int>();
            bool[] visited = new bool[graph.Nodes.Length];
            int source = graph.Source;

            queue.Enqueue(source);
            visited[source] = true;

            while (queue.Count > 0)
            {
                int currentNode = queue.Dequeue();
                reachableFromSource.Add(currentNode);

                foreach (Edge edge in graph.Nodes[currentNode].Edges)
                {
                    if (!visited[edge.To] && edge.Capacity - edge.Flow > 0)
                    {
                        queue.Enqueue(edge.To);
                        visited[edge.To] = true;
                    }
                }
            }

            // Updating Mask after classification
            for (int y = 0; y < mask.Rows; y++)
            {
                for (int x = 0; x < mask.Cols; x++)
                {
                    int nodeIndex = y * mask.Cols + x;
                    if (reachableFromSource.Contains(nodeIndex))
                    {
                        if (y % 50 == 0 && x % 50 == 0)
                            Console.WriteLine("MinCut Updating xy "+ x+","+y);
                        mask.Set<byte>(y, x, (byte)GrabCutClasses.PR_FGD);
                    }
                    else
                    {
                        mask.Set<byte>(y, x, (byte)GrabCutClasses.PR_BGD);
                    }
                }
            }
        }
    }
}
