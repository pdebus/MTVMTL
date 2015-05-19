#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"

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

	typedef Data< mf_t, 2> data_t;	
	data_t myData=data_t();
	myData.rgb_imread(argv[1]);
	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	func_t myFunc(3.0, myData);

	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;
	tvmin_t myTVMin(myFunc, myData);


	return 0;
}
