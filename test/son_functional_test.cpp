#include <iostream>

#define TV_DATA_DEBUG
#define TV_FUNC_DEBUG_VERBOSE
#define TVMTL_TVMIN_DEBUG_VERBOSE

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"

int main(int argc, const char *argv[])
{
	using namespace tvmtl;
/*
	if (argc != 2){
	    std::cerr << "Usage : " << argv[0] << " image" << std::endl;
	    return 1;
	}
*/
	typedef Manifold< SO, 3 > mf_t;
	typedef Data< mf_t, 2> data_t;	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;

	data_t myData = data_t();

	myData.create_nonsmooth_son(30,35);
	myData.output_matval_img("son_img.csv");

	double lam=0.1;
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-10);

	func_t::result_type result = 0.0;
	
	//Functional evaluation
	result = myFunc.evaluateJ();
	//Gradient
	myFunc.evaluateDJ();
	//Hessian
	myFunc.evaluateHJ();

	return 0;
}
