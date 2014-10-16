#ifndef BACKPROP_HPP
#define BACKPROP_HPP

#include <math.h>

class BackProp
{

    public:
	BackProp(int nl, int *sz, double b, double a=0.0);
	~BackProp();

	//back-propogates error for one set of inout
	void bpgt(double *in, double *tgt);

	//feeds forward activations for one set of input
	void ffwd(double *in);

	//returns mean sqaure error of the net
	double mse(double *tgt) const;

	//returns ith output of the net
	double Out(int i) const;

    private:
	inline double sigmoid(double in){return 1.0/(1.0+exp(-in));}
	double **out;//output of each neuron
	double **delta;//delta error for each neuron
	double ***weight;//vector of weights for each neuron
	int numl;//no. of layers in net including input layer
	double beta;//learning rate
	double alpha;//momentum paremter
	double ***prevDwt;//storage for each weight change made in previus epoch	
	int *lsize;//vector of numl elements for size of each layer
};

#endif//BACKPROP_HPP
