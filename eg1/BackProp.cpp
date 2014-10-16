#include <BackProp.hpp>
#include <stdlib.h>
#include <time.h>

BackProp::BackProp(int nl,int *sz, double b, double a):beta(b),alpha(a)
{
    //set number of layers and their sizes
    numl=nl;
    lsize=new int[numl];    
    for(int i=0; i<numl;++i)
    {
		lsize[i]=sz[i];
    }

    //allocate memory for output of each neuron
    out = new double*[numl];
    for(int i=0;i<numl;++i)
    {
		out[i]=new double[lsize[i]];
    }

    //allocate memory for delta
    delta = new double *[numl];
    for(int i=1;i<numl;++i)
    {
		delta[i]=new double[lsize[i]];
    }

    //allocate memory for weights
    weight = new double **[numl];
    for(int i=1;i<numl;++i)
    {
		weight[i]= new double*[lsize[i]];
		for(int j=0;j<lsize[i];++j)
		{
	    	weight[i][j]=new double[lsize[i-1]+1];
		}
    }

    //allocate memory for previous weights
    prevDwt= new double **[numl];
    for(int i=1;i<numl;++i)
    {
		prevDwt[i]=new double *[lsize[i]];
		for(int j=0;j<lsize[i];++j)
		{
	    	prevDwt[i][j]=new double[lsize[i-1]+1];
    	}
    }

    //seed and assign random weights
    srand((unsigned)(time(NULL)));
    for(int i=1;i<numl;++i)
    {
		for(int j=0;j<lsize[i];++j)
		{
	    	for(int k=0; k< lsize[i-1]+1;++k)
	    	{
				//32767
				weight[i][j][k]= rand()/(RAND_MAX/2)-1;
				//initialise previous weights to 0 for first iteration
				prevDwt[i][j][k]=0.0;
	    	}
		}
    }

}	

BackProp::~BackProp()
{
    for(int i=0;i<numl;++i)
    {
		delete[] out[i];
    }
    delete[] out;

    for(int i=1;i<numl;++i)
    {                 
    	for(int j=0;j<lsize[i];++j)
    	{
	    	delete[] weight[i][j];
	    	delete[] prevDwt[i][j];
		}
        delete[] weight[i];
		delete[] prevDwt[i];
		delete[] delta[i];
    }
    delete[] delta;
    delete[] weight;
    delete[] prevDwt;
    delete[] lsize;
}

//mean square error
double BackProp::mse(double *tgt) const
{
    double mse = 0.0;
    for(int i=0; i<lsize[numl-1];++i)
    {
    	mse += (tgt[i] - out[numl-1][i]) * (tgt[i] - out[numl-1][i]);
    }
    return mse/2.0;
}

//returns ith output of the net
double BackProp::Out(int i) const
{
    return out[numl-1][i];
}

//feed forward one set of input
void BackProp::ffwd(double *in)
{
    double sum;

    //assign content to inner layer
    for(int i=0; i<lsize[0];++i)
    {
		out[0][i]=in[i];//output from_from_neuron(i,j) Jth neuron in Ith layer
    }

    //asign output(activation) value to each neuron using sigmoid func
    for(int i=1;i<numl;++i)//for each layer
    {
		for(int j=0;j<lsize[i];++j)//for each neuron in current layer
		{
	    	sum  =0.0;
    	    for(int k = 0; k<lsize[i-1];++k)//for input from each neuron in the preceding layer
	    	{
				sum+=out[i-1][k]*weight[i][j][k];//Apply weight to inputs and add to sum
	    	}
	    	sum+=weight[i][j][lsize[i-1]];;//apply bias
	    	out[i][j]=sigmoid(sum);//apply sigmoid function
		}
    }
}

//backpropogate errors fromouput layer uptill the first hidden layer
void BackProp::bpgt(double *in, double *tgt)
{
    double sum;

    //update output values for each neuron
    ffwd(in);

     //find delta for outpout layer
    for(int i=0;i<lsize[numl-1];++i)
    {
		delta[numl-1][i]=out[numl-1][i] * (1.0-out[numl-1][i]) * (tgt[i]-out[numl-1][i]);
    }

    //find delta forr hidden layers
    for(int i=numl-2;i>0;--i)
    {
		for(int j=0; j<lsize[i];++j)
		{
	    	sum = 0.0;
	    	for(int k=0;k<lsize[i+1];k++)
	    	{
				sum+=delta[i+1][k]*weight[i+1][k][j];
	    	}
	    	delta[i][j]=out[i][j] * (1.0-out[i][j]) * sum;
		}
    }

    //apply momentum (does nothing if alpha  =0)
    for(int i=1;i<numl;++i)
    {
		for(int j=0;j<lsize[i];++j)
		{
	    	for(int k=0; k< lsize[i-1];++k)
		    {
				weight[i][j][k]+=alpha * prevDwt[i][j][k];
		    }
		    weight[i][j][lsize[i-1]]+=alpha * prevDwt[i][j][lsize[i-1]];
		}
    }

    //adjust weights using steepest descent
    for(int i=1;i<numl;++i)
    {
		for(int j=0; j<lsize[i];++j)
		{
		    for(int k=0;k<lsize[i-1];++k)
		    {
				prevDwt[i][j][k] = beta * delta[i][j] * out[i-1][k];
				weight[i][j][k] += prevDwt[i][j][k];
	    	}
		    prevDwt[i][j][lsize[i-1]] = beta * delta[i][j];
		    weight[i][j][lsize[i-1]] += prevDwt[i][j][lsize[i-1]];
		}
    }
}

    


   



   


