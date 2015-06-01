#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#define TV_FUNC_DEBUG
#include "../core/data.hpp"
#include "../core/functional.hpp"

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>


int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	if (argc != 2){
	    std::cerr << "Usage : " << argv[0] << " image" << std::endl;
	    return 1;
	}

	typedef Manifold< EUCLIDIAN, 1 > mf_t;
	typedef typename mf_t::value_type vec1d;

	typedef Data< mf_t, 2> data_t;	
	typedef typename data_t::storage_type store_type;

	data_t myData=data_t();
	
	myData.rgb_readBrightness(argv[1]);
	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	func_t myFunc(3.0, myData);

	func_t::result_type result = 0.0;
	
	//Functional evaluation
	std::cout << "Evaluate Functional..." << std::endl;
	result = myFunc.evaluateJ();
	std::cout << "Functional evaluation for Picture " << argv[1] << ": " << result << std::endl;
	//Gradient
	std::cout << "Evaluate Gradient..." << std::endl;
	myFunc.evaluateDJ();
	//Hessian
	std::cout << "Evaluate Hessian..." << std::endl;
	myFunc.evaluateHJ();


	return 0;
}
