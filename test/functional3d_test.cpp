#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#define TV_FUNC_DEBUG
#define TV_FUNC_DEBUG_VERBOSE
#include <mtvmtl/core/data.hpp>
#include <mtvmtl/core/functional.hpp>

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>


int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	typedef Manifold< EUCLIDIAN, 1 > mf_t;
	typedef typename mf_t::value_type vec3d;

	typedef Data< mf_t, 3> data_t;	
	typedef typename data_t::storage_type store_type;

	data_t myData=data_t();
	
	myData.create_noisy_gray(4, 3, 2);
	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t, 3> func_t;
	func_t myFunc(0.1, myData);

	func_t::result_type result = 0.0;
	
	//Functional evaluation
	result = myFunc.evaluateJ();
	//Gradient
	myFunc.evaluateDJ();
	//Hessian
	myFunc.evaluateHJ();

	std::cout << "Functional evaluation for Picture " << result << std::endl;

	return 0;
}
