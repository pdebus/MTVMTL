#include <iostream>
#include <opencv2/highgui/highgui.hpp>

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

	typedef Manifold< EUCLIDIAN, 3 > mf_t;
	typedef typename mf_t::value_type vec3d;

	typedef Data< mf_t, 2> data_t;	
	typedef typename data_t::storage_type store_type;

	data_t myData=data_t();
	
	myData.rgb_imread(argv[1]);
	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	func_t myFunc(3.0, myData);

	func_t::return_type result = 0.0;
	    
	result = myFunc.evaluateJ();
	myFunc.evaluateDJ();

	std::cout << "Functional evaluation for Picture " << argv[1] << ": " << result << std::endl;

	return 0;
}
