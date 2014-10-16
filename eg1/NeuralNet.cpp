#include <BackProp.hpp>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    //prepare XOR training data
    double data[][4] =
    {
	0,0,0,0,
	0,0,1,1,
	0,1,0,1,
	0,1,1,0,
	1,0,0,1,
	1,0,1,0,
	1,1,0,0,
	1,1,1,1
    };

    //prepare test data
    double testData[][3] =
    {
	0,0,0,
	0,0,1,
	0,1,0,
	0,1,1,
	1,0,0,
	1,0,1,
	1,1,0,
	1,1,1
    };

    //defining a net with 4 layers having 3,3,3, and 1 neuorn repectively.
    //The first layer is the input layer i.e. simply a holder for the input parameters
    //and has top be the same as the number of input parameters, 3 in oiur example.
    int numLayers = 4, lsz[4] = {3,3,2,1};

    //Learning rate = beta
    //momentum = alpha
    //Threshold = thresh (value of target mse - training stops once it is achieved)
    double beta = 0.3, alpha = 0.1, Thresh = 0.00001;

    //max. number of iterations during training
    long num_iter = 2000000;
   
    //Create the net:
    BackProp *bp = new BackProp(numLayers, lsz, beta, alpha);

   cout << endl << "Now training the network..." << endl;

    long i =0;
    for(i=0; i<num_iter; ++i)
    {
		bp->bpgt(data[i%8], &data[i%8][3]);
	
		if(bp->mse(&data[i%8][3]) < Thresh)
		{
	    	cout << endl << "Network trained.  Threshold value achieved in " << i << " iterations." << endl;
		    cout << "MSE: " << bp->mse(&data[i%8][3]) << endl << endl;
		    break;
		}

		if( i%(num_iter/10) == 0)
		{
	    	cout << endl << "MSE: " << bp->mse(&data[i%8][3]) << " ...still training" << endl;
		}
    }

    if(i == num_iter)
    {
		cout << endl << i << " iterations completed, threshold not reached! MSE: " << bp->mse(&data[(i-1)%8][3]) << endl;
    }

    cout << "Now using trained network to  make predictions on test data..." << endl << endl;

    for(i=0; i<8;++i)
    {
		bp->ffwd(testData[i]);
    	cout << testData[i][0] << "  " << testData[i][1] << "  " << testData[i][2] << "  " << bp->Out(0) << endl;
    }

    return 0;
}

	




