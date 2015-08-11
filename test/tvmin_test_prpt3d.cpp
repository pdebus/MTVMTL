#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>


//#define TVMTL_TVMIN_DEBUG
//#define TVMTL_TVMIN_DEBUG_VERBOSE
//#define TV_DATA_DEBUG


#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"

#include <vpp/vpp.hh>

using namespace tvmtl;

typedef Manifold< EUCLIDIAN, 1 > mf_t;
typedef Data< mf_t, 3> data_t;	
typedef Functional<FIRSTORDER, ANISO, mf_t, data_t, 3> func_t;
typedef TV_Minimizer< PRPT, func_t, mf_t, data_t, OMP, 3 > tvmin_t;


int main(int argc, const char *argv[])
{


	double lam=0.1;

	data_t myData=data_t();
	myData.create_noisy_gray(7,5,4);

	func_t myFunc(lam, myData);
	myFunc.seteps2(0.0);

	tvmin_t myTVMin(myFunc, myData);
	myTVMin.use_approximate_mean(false);
    
	int ns = myData.img_.nslices();  // z
	int nr = myData.img_.nrows();    // y
	int nc = myData.img_.ncols();    // x
    
    
	for(int s = 0; s < ns; ++s){
	    std::cout << "\nSlice " << s << ":\n";
	    for(int r = 0; r < nr; ++r){
		for(int c = 0; c < nc; ++c)
		    std::cout << myData.img_(s, r, c) << " ";
		std::cout << std::endl;
	    }
	}

	myTVMin.minimize();
	
	for(int s = 0; s < ns; ++s){
	    std::cout << "\nSlice " << s << ":\n";
	    for(int r = 0; r < nr; ++r){
		for(int c = 0; c < nc; ++c)
		    std::cout << myData.img_(s, r, c) << " ";
		std::cout << std::endl;
	    }
	}		


	return 0;
}
