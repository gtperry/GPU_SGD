using System;

using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
//using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
namespace GPU_SGD
{
	
    public class GPU_SGD
    {
    	Context context;
	    Device dev;
	    Accelerator accelerate;
        double[,] arr;
        double[] weights;
        double[] y;
    	public GPU_SGD(double[,] array, double[] y)
        {
           	this.arr = array;
            this.y = y;
            //this.distributions = new int[10];
            this.context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());

            this.dev = this.context.GetPreferredDevice(preferCPU: false);
            

        }
        public void SGDfit(int epoch, double learning_rate=0.1){
            return;
        }
        public void fit(int epochs, double learning_rate=0.01){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            this.weights = new double[this.arr.GetLength(1)];
            this.weights = InitializeGaussian(this.weights);
            print1d(this.weights);
            var XBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(this.arr.GetLength(0),this.arr.GetLength(1)));
            var YBuffer = accelerate.Allocate1D<double>(new Index1D(this.y.GetLength(0)));
            var WeightsBuffer = accelerate.Allocate1D<double>(new Index1D(this.weights.GetLength(0)));
            var gradBuffer = accelerate.Allocate1D<double>(new Index1D(this.weights.GetLength(0)));
            var hBuffer = accelerate.Allocate1D<double>(new Index1D(this.weights.GetLength(0)));


            XBuffer.CopyFromCPU(this.arr);
            YBuffer.CopyFromCPU(this.y);
            WeightsBuffer.CopyFromCPU(this.weights);
            var setBuffToValueKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(setBuffToValueKernal);
            var GradientKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> , int
                >(GradientKernal);
            var ThetaKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense> ,
                double,
                int>(ThetaKernal);
            for(int i = 0; i < epochs; i++){
                setBuffToValueKern(gradBuffer.Extent.ToIntIndex(), gradBuffer.View, 0.0);
                GradientKern(XBuffer.Extent.ToIntIndex(), XBuffer.View, WeightsBuffer.View, gradBuffer.View, YBuffer.View, this.arr.GetLength(0));
                accelerate.Synchronize();
                ThetaKern(WeightsBuffer.Extent.ToIntIndex(), WeightsBuffer.View, gradBuffer.View, learning_rate, this.y.GetLength(0));

            }
            this.weights = WeightsBuffer.GetAsArray1D();


        }
        static double[] BatchGradientDescentADJ(double[,] X, double[] y, double alpha, int epochs)
        {
            int m = X.GetLength(0);
            int n = X.GetLength(1);
            double[] theta = new double[n];
            double[] grad = new double[n];
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double[] h = new double[m];
                for (int i = 0; i < m; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < n; j++)
                    {
                        sum += (X[i, j] * theta[j])- (y[i]/n) ;
                    }
                    h[i] = sum;
                    
                }

                // double[] error = new double[m];
                // for (int i = 0; i < m; i++)
                // {
                //     error[i] = h[i] - y[i];
                // }

                for (int i = 0; i < n; i++)
                {
                    double sum1 = 0;
                    for (int j = 0; j < m; j++)
                    {
                        sum1 += h[j] * X[j, i]/m;
                    }
                    grad[i] = sum1;
                }

                for (int j = 0; j < n; j++)
                {
                    theta[j] = theta[j] - alpha * grad[j];
                }
            }
            return theta;
        }
        public double[] StochasticGradientDescent(double[,] X, double[] Y, double learningRate, int epochs, int batch){
            int numFeatures = X.GetLength(1);
            double[] weights = new double[numFeatures];
            int index;
            Random rand = new Random();

            // Train the model
            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < batch; j++)
                {
                    // Compute the prediction
                    index = rand.Next(X.GetLength(0));
                    double prediction = 0 ;
                    for (int k = 0; k < numFeatures; k++)
                    {
                        prediction += weights[k] * X[index,k];
                    }
                    // Console.Write("Prediction: ");
                    // Console.WriteLine(prediction);
                    // Console.ReadLine();
                    // Compute the error
                    double error =  Y[index] -  prediction ;
                    // Console.Write("Error: ");
                    // Console.WriteLine(error);
                    // Console.ReadLine();
                    // Update the weights
                    //weights[0] += learningRate * error * X[index,0] ;
                    for (int k = 0; k < numFeatures; k++)
                    {
                        weights[k] += learningRate * (error * X[index,k])/numFeatures;
                    }
                    // Console.Write("Weights[0]: ");
                    // Console.WriteLine(weights[0]);
                    // Console.ReadLine();
                }
            }
            return weights;
        }
        public double[] StochasticGradientDescentADJ(double[,] X, double[] Y, double learningRate, int epochs, int batch){
            int numFeatures = X.GetLength(1);
            double[] weights = new double[numFeatures];
            int index;
            Random rand = new Random();

            // Train the model
            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < batch; j++)
                {
                    // Compute the prediction
                    index = rand.Next(X.GetLength(0));
                    double error = 0 ;
                    for (int k = 0; k < numFeatures; k++)
                    {
                        error += (Y[index]/numFeatures) - (weights[k] * X[index,k]);
                    }
                    // Console.Write("Prediction: ");
                    // Console.WriteLine(prediction);
                    // Console.ReadLine();
                    // Compute the error
                    //double error =  Y[index] -  prediction ;
                    // Console.Write("Error: ");
                    // Console.WriteLine(error);
                    // Console.ReadLine();
                    // Update the weights
                    //weights[0] += learningRate * error * X[index,0] ;
                    for (int k = 0; k < numFeatures; k++)
                    {
                       

                        weights[k] += learningRate * (error * X[index,k])/numFeatures;
                    }
                    // Console.Write("Weights[0]: ");
                    // Console.WriteLine(weights[0]);
                    // Console.ReadLine();
                }
            }
            return weights;
        }
        //GPU SGD final
        public double[] SGDgpu(double[,] X, double[] Y, double learningRate, int epochs, int batch){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            
            //this.weights = InitializeGaussian(this.weights);
            //print1d(this.weights);\
            int m = X.GetLength(0);
            int n = X.GetLength(1);
            double[] weights = new double[n];

            var XBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(m,n));
            XBuffer.CopyFromCPU(X);
            var YBuffer = accelerate.Allocate1D<double>(new Index1D(m));
            YBuffer.CopyFromCPU(Y);
            var WeightsBuffer = accelerate.Allocate1D<double>(new Index1D(n));
            WeightsBuffer.CopyFromCPU(weights);
            
            var errorBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            
            int index;
            Random rand = new Random();

            var part1 = accelerate.LoadAutoGroupedStreamKernel<
                Index1D, 
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, 
                int,
                int>(batchRunPart1Kernal);
            var part2 = accelerate.LoadAutoGroupedStreamKernel<
                Index1D, 
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, 
                int,
                int,
                double>(batchRunPart2Kernal);
            var setBuffToValueKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(setBuffToValueKernal);


            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < batch; j++)
                {
                    index = rand.Next(m);
                    setBuffToValueKern(errorBuffer.Extent.ToIntIndex(), errorBuffer.View, 0.0);
                    part1(WeightsBuffer.Extent.ToIntIndex(), XBuffer.View, WeightsBuffer.View, YBuffer.View, errorBuffer.View, index, n );
                    part2(WeightsBuffer.Extent.ToIntIndex(), XBuffer.View, WeightsBuffer.View, errorBuffer.View, index, n, learningRate);
                }
            }
            return WeightsBuffer.GetAsArray1D();
        }
        static void batchRunPart1Kernal(Index1D index, 
            ArrayView2D<double, Stride2D.DenseX> xView,
            ArrayView1D<double, Stride1D.Dense> weightView, 
            ArrayView1D<double, Stride1D.Dense> yView, 
            ArrayView1D<double, Stride1D.Dense> errorView, 

            int randind,

            int numfeatures)
        {
            
            Atomic.Add(ref errorView[new Index1D(0)], (yView[randind]/numfeatures) - (weightView[index] *xView[new Index2D(randind, index.X)]));
        }
        static void batchRunPart2Kernal(Index1D index, 
            ArrayView2D<double, Stride2D.DenseX> xView,
            ArrayView1D<double, Stride1D.Dense> weightView, 
            ArrayView1D<double, Stride1D.Dense> errorView, 

            int randind,

            int numfeatures,
            double lr)
        {
            
            Atomic.Add(ref weightView[index], lr*(errorView[new Index1D(0)]*xView[new Index2D(randind,index.X)])/numfeatures);
        }
        public double[] BatchGradientDescent(double[,] X, double[] y, double alpha, int epochs)
        {
            int m = X.GetLength(0);
            int n = X.GetLength(1);
            double[] theta = new double[n];
            double[] grad = new double[n];
            //Console.WriteLine("BATCH");
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double[] h = new double[m];
                for (int i = 0; i < m; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < n; j++)
                    {
                        sum += X[i, j] * theta[j];
                    }
                    h[i] = sum;
                }

                double[] error = new double[m];
                for (int i = 0; i < m; i++)
                {
                    error[i] = h[i] - y[i];
                }
                // Console.WriteLine("Error");
                // print1d(error);

                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < m; i++)
                    {
                        sum += error[i] * X[i, j];
                    }
                    grad[j] = sum / m;
                }
                // Console.WriteLine("Grad");
                // print1d(grad);
                for (int j = 0; j < n; j++)
                {
                    theta[j] = theta[j] - alpha * grad[j];
                }
                // Console.WriteLine("Theta");
                // print1d(theta);
            }
            return theta;
        }
        public void fitv2(int epochs, double learning_rate=0.01){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            this.weights = new double[this.arr.GetLength(1)];
            //this.weights = InitializeGaussian(this.weights);
            //print1d(this.weights);\
            int m = this.arr.GetLength(0);
            int n = this.arr.GetLength(1);
            var XBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(this.arr.GetLength(0),this.arr.GetLength(1)));
            var YBuffer = accelerate.Allocate1D<double>(new Index1D(this.y.GetLength(0)));
            var WeightsBuffer = accelerate.Allocate1D<double>(new Index1D(this.weights.GetLength(0)));
            var gradBuffer = accelerate.Allocate1D<double>(new Index1D(this.weights.GetLength(0)));
            var errorBuffer = accelerate.Allocate1D<double>(new Index1D(this.arr.GetLength(0)));


            XBuffer.CopyFromCPU(this.arr);
            YBuffer.CopyFromCPU(this.y);
            WeightsBuffer.CopyFromCPU(this.weights);
            var setBuffToValueKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(setBuffToValueKernal);
            var GradientKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> , int
                >(GradKernal);
            var ThetaKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense> ,
                double,
                int>(ThetaKernal);

            var errorKern = accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> ,
                int>(errorKernal);
            
            
            for(int i = 0; i < epochs; i++){
                setBuffToValueKern(gradBuffer.Extent.ToIntIndex(), gradBuffer.View, 0.0);
                setBuffToValueKern(errorBuffer.Extent.ToIntIndex(), errorBuffer.View, 0.0);
                errorKern(XBuffer.Extent.ToIntIndex(), XBuffer.View, WeightsBuffer.View, errorBuffer.View, YBuffer.View, n);
                // Console.WriteLine("errorBuffer");
                // print1d(errorBuffer.GetAsArray1D());
                GradientKern(XBuffer.Extent.ToIntIndex(), XBuffer.View, errorBuffer.View, gradBuffer.View, m);
                // Console.WriteLine("gradBuffer");
                // print1d(gradBuffer.GetAsArray1D());
                //accelerate.Synchronize();
                ThetaKern(WeightsBuffer.Extent.ToIntIndex(), WeightsBuffer.View, gradBuffer.View, learning_rate,m);
                // Console.WriteLine("WeightsBuffer");
                // print1d(WeightsBuffer.GetAsArray1D());

            }
            this.weights = WeightsBuffer.GetAsArray1D();


        }
        //Non GPU Sgd
        public void SGDfit(int epochs, int batch, double learning_rate=0.01){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            this.weights = new double[this.arr.GetLength(1)];
            //this.weights = InitializeGaussian(this.weights);
            //print1d(this.weights);\
            int m = this.arr.GetLength(0);
            int n = this.arr.GetLength(1);
            Index2D sub2d = new Index2D(batch, n);
            Index1D mInd = new Index1D(m);
            Index1D nInd = new Index1D(n);
            Index1D batchInd = new Index1D(batch);

            var XBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(m,n));
            //var SubXBuffer = accelerate.Allocate2DDenseX<double>(sub2d);

            var YBuffer = accelerate.Allocate1D<double>(mInd);
            //var SubYBuffer = accelerate.Allocate1D<double>(batchInd);
            var WeightsBuffer = accelerate.Allocate1D<double>(nInd);
            var gradBuffer = accelerate.Allocate1D<double>(nInd);
            var errorBuffer = accelerate.Allocate1D<double>(mInd);
            //var SuberrorBuffer = accelerate.Allocate1D<double>(batchInd);

            Random rand = new Random();

            XBuffer.CopyFromCPU(this.arr);
            YBuffer.CopyFromCPU(this.y);
            WeightsBuffer.CopyFromCPU(this.weights);
            var setBuffToValueKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(setBuffToValueKernal);
            var GradientKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> , int
                >(GradKernal);
            var ThetaKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense> ,
                double,
                int>(ThetaKernal);

            var errorKern = accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> ,
                int>(errorKernal);
            //print2d(XBuffer.View.SubView((0,0), (batch,n)).GetAsArray2D());
            //Console.ReadLine();
            
            for(int i = 0; i < epochs; i++){
                int index = rand.Next(m-batch);
                
                var SubXBuffer = XBuffer.View.SubView((index,0), (batch,n));
                var SuberrorBuffer = errorBuffer.View.SubView(index, batch);
                setBuffToValueKern(gradBuffer.Extent.ToIntIndex(), gradBuffer.View, 0.0);
                setBuffToValueKern(errorBuffer.Extent.ToIntIndex(), errorBuffer.View, 0.0);
                //errorKern(sub2d, XBuffer.View.SubView((index,0), (batch,n)), WeightsBuffer.View, errorBuffer.View.SubView(index, batch), YBuffer.View.SubView(index, batch), n);
                errorKern(sub2d, SubXBuffer, WeightsBuffer.View, SuberrorBuffer, YBuffer.View.SubView(index, batch), n);

                    // Console.WriteLine("errorBuffer");
                    // print1d(errorBuffer.GetAsArray1D());
                GradientKern(sub2d, SubXBuffer, SuberrorBuffer, gradBuffer.View, batch);
                    // Console.WriteLine("gradBuffer");
                    // print1d(gradBuffer.GetAsArray1D());
                    //accelerate.Synchronize();
                ThetaKern(WeightsBuffer.Extent.ToIntIndex(), WeightsBuffer.View, gradBuffer.View, learning_rate,batch);
                
                
                // Console.WriteLine("WeightsBuffer");
                // print1d(WeightsBuffer.GetAsArray1D());

            }
            this.weights = WeightsBuffer.GetAsArray1D();


        }
        public double[] getWeights(){
            return this.weights;
        }
        public double[] predict(double[,] inputs){
            double[] outputs = new double[inputs.GetLength(0)];
            for(int i = 0; i < inputs.GetLength(0); i ++){
                outputs[i] = 0;
                for(int j = 0; j < inputs.GetLength(1); j++){
                    outputs[i] += inputs[i,j] * this.weights[j];
                }
            }
            return outputs;

        }
        static void updateWeights(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> weightView, 
            ArrayView1D<double, Stride1D.Dense> gradView, 

            double learning_rate)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            weightView[index] -= learning_rate * gradView[index];
        }
        static void setBuffToValueKernal(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> buff, 
            double setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void MatrixMultiply2DKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            ///<summary> Does Matrix Multiplication on two arrayviews, and then stores in a new arrayview </summary>
            ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
            ///<param name="aView">(ArrayView2D<float, Stride2D.DenseX>) 1st ArrayView being multiplied</param>
            ///<param name="bView">(ArrayView2D<float, Stride2D.DenseX>) 2nd ArrayView being multiplied</param>
            ///<param name="cView">(ArrayView2D<float, Stride2D.DenseX>) Buffer where new value goes</param>
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;
            for (var i = 0; i < aView.IntExtent.Y; i++)
                sum += aView[new Index2D(x, i)] * bView[new Index2D(i, y)];

            cView[index] = sum;
        }
        //Index3D in format (epoch, batch)
        // static void SGDKernal(Index2D index,
        //     ArrayView2D<float, Stride2D.DenseX> inputView,
        //     ArrayView1D<double, Stride1D.Dense>  weightView,
            

        //     ArrayView1D<double, Stride1D.Dense>  yView,
            

        //     double alpha,
        //     int features,
        //     int inputlength

        //     ){
        //     double slope = 0;
        //     double intercept = 0;
        //     double gradient = 0;
        //     double diff = 0;
        //     for(int i = 0; i < features; i++){
        //         diff += weightView[i] * inputView[randind, i];  
        //     }
        //     diff = diff - yView[randind];
        //     weightView[index] = weightView[index] - (alpha * );
        // }
        //Index2D format(row, columns)
        static void GradientKernal(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> inputView,
            ArrayView1D<double, Stride1D.Dense>  thetaView,
            ArrayView1D<double, Stride1D.Dense>  gradView,
            ArrayView1D<double, Stride1D.Dense>  yView,
            int m
            
            ){
            
            Atomic.Add(ref gradView[index.Y], (((inputView[index] * thetaView[index.Y]) - yView[index.X]   ) * inputView[index]));
        }
        static void errorKernal(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> inputView,
            ArrayView1D<double, Stride1D.Dense>  thetaView,
            ArrayView1D<double, Stride1D.Dense>  errorView,
            ArrayView1D<double, Stride1D.Dense>  yView,
            int n
            
            ){
            
            Atomic.Add(ref errorView[index.X], ((inputView[index] * thetaView[index.Y]) - (yView[index.X]/n)) );
        }
        static void GradKernal(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> inputView,
            ArrayView1D<double, Stride1D.Dense>  errorView,
            ArrayView1D<double, Stride1D.Dense>  gradView,
            int m
            
            ){
            
            Atomic.Add(ref gradView[index.Y], (errorView[index.X] * inputView[index])/m);
        }
        static void ThetaKernal(Index1D index, 
            ArrayView1D<double, Stride1D.Dense>  thetaView,
            ArrayView1D<double, Stride1D.Dense>  gradView,
            double alpha,
            int ylength){
            Atomic.Add(ref thetaView[index], (-1.0*((alpha *gradView[index]))));

        }
        void print1d(double[] array)
        {
            Console.Write("[");
            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0}, ", array[j]);
            }
            Console.WriteLine("]");

        }
        void print2d(double[,] array)
        {
            Console.WriteLine(array);

            for (int i = 0; i < array.GetLength(0); i++)
            {
                Console.Write("[");
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write("{0}, ", array[i, j]);
                }
                Console.Write("]");
                Console.WriteLine(", ");
            }
            Console.WriteLine("]");
        }
        
        public static double[] InitializeGaussian(double[] weights) {
            Random random = new Random();
            double mean = 0.0;
            double stdDev = 0.01;
            for (int i = 0; i < weights.GetLength(0); i++) {
                double u1 = 1.0 - random.NextDouble(); // uniform(0,1] random doubles
                double u2 = 1.0 - random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                       Math.Sin(2.0 * Math.PI * u2); // random normal(0,1)
                double randGaussian = mean + stdDev * randStdNormal; // random normal(mean,stdDev^2)
                weights[i] = randGaussian;
            }
            return weights;
        }
        public double compareWeights(double[] w1, double[] w2){
            double sum = 0.0;
            for(int i = 0; i < w1.GetLength(0); i++){
                sum+= Math.Abs(w1[i] - w2[i]);
            }
            return sum/w1.GetLength(0);

        }
        static void Main(string[] args)
        {
            // double[,] a = new[,] {{4.2}, {5.43}, {3.221}, {7.34235}, {1.931}, {1.2}, {5.43}, {8.0}, {7.34235}, {1.931}};
            // double[] b = new double[a.GetLength(0)];
            // for (int i = 0; i < b.GetLength(0); i++) {
            //   b[i] = (a[i,0] * .498);
            // }
            int numRows = 100000; // number of rows in the 2D array
            int numCols = 1000; // number of columns in the 2D array
            double[,] x = new double[numRows, numCols];
            double[] y = new double[numRows];
            double[] w = new double[numCols];

            Random rand = new Random(); // create a random number generator

            // fill the 2D array x with random values
            for (int i = 0; i < numRows; i++)
            {

                for (int j = 0; j < numCols; j++)
                {
                    x[i, j] = rand.NextDouble(); // assign a random value to the current element

                }
            }
            for (int i = 0; i< numCols; i++){
                w[i] = rand.NextDouble() *10;
            }
            for (int i = 0; i < numRows; i++)
            {
                y[i] = 0;
                for (int j = 0; j < numCols; j++)
                {
                    y[i] += x[i, j] * w[j]; // assign a random value to the current element

                }
            }

            Stopwatch stop = new Stopwatch();
            GPU_SGD sgd = new GPU_SGD(x,y);
            // stop.Start();
            // sgd.SGDfit(100,1000, 0.00001);
            // stop.Stop();
            // Console.Write("Elapsed time for GPU: ");
            // Console.WriteLine(stop.ElapsedMilliseconds);
            // stop.Reset();
            // Console.WriteLine("Here");
            // Console.Write("GPU average error: ");
            // Console.WriteLine(sgd.compareWeights(w,sgd.getWeights()));

            // //sgd.print1d(sgd.getWeights());
            // Console.WriteLine("_________________");
            // Console.WriteLine("Real Weights:");
            //sgd.print1d(w);

            Console.WriteLine("_________________");
            // // stop.Start();
            // // //sgd.print1d(sgd.BatchGradientDescent(x,y,0.1,1000));
            // // stop.Stop();
            // // Console.Write("Elapsed time for Reg: ");
            // // Console.WriteLine(stop.ElapsedMilliseconds);
            // // stop.Reset();
            // // Console.WriteLine("_________________");
            // //             stop.Start();
            // Console.Write("ADJ average error: ");
            // Console.WriteLine(sgd.compareWeights(w,sgd.BatchGradientDescent(x,y,0.001,1000)));
            // // //sgd.print1d(BatchGradientDescentADJ(x,y,0.1,1000));
            // // stop.Stop();
            // // Console.Write("Elapsed time ADJ: ");
            // // Console.WriteLine(stop.ElapsedMilliseconds);
            // stop.Reset();

            stop.Start();
            //sgd.print1d(sgd.StochasticGradientDescent(x,y,0.1,1000, 10000));
            Console.Write("SGD weights: ");
            //sgd.print1d(sgd.StochasticGradientDescentADJ(x,y,0.1,1000, 1000));
            Console.WriteLine(sgd.compareWeights(w,sgd.StochasticGradientDescentADJ(x,y,0.01,10000, 1000)));

            Console.WriteLine("_________________");
            stop.Stop();
            Console.Write("Elapsed time SGD: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();

            stop.Start();
            //sgd.print1d(sgd.StochasticGradientDescent(x,y,0.1,1000, 10000));
            Console.Write("GPU weights: ");
            //sgd.print1d(sgd.SGDgpu(x,y,0.1,1000, 1000));
            Console.WriteLine(sgd.compareWeights(w,sgd.SGDgpu(x,y,0.01,10000, 1000)));
            stop.Stop();
            Console.Write("Elapsed time GPU: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();
            //BatchGradientDescentADJ

        }

    }
}
