#include <iostream>


//#define TVMTL_TVMIN_DEBUG
//#define TV_SON_DEBUG
//#define TV_DATA_DEBUG
//#define TV_FUNC_DEBUG_VERBOSE
//#define TVMTL_TVMIN_DEBUG_VERBOSE
//#define TV_VISUAL_DEBUG

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"
#include "../core/visualization.hpp"

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
	typedef Functional<FIRSTORDER,ANISO, mf_t, data_t> func_t;
	typedef TV_Minimizer< PRPT, func_t, mf_t, data_t, OMP > tvmin_t;
	typedef Visualization<SO, 3, data_t> visual_t;

	data_t myData = data_t();

	myData.create_nonsmooth_son(30,30);
	myData.add_gaussian_noise(0.2);
	myData.output_matval_img("son_img1.csv");

	double lam=0.2;
	func_t myFunc(lam, myData);
	myFunc.seteps2(0);

	tvmin_t myTVMin(myFunc, myData);

	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();


	visual_t myVisual(myData);
	myVisual.saveImage("denoised_son_prpt.png");
	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SO(N) Cubes Visualization 30x30 ");
	std::cout << "Rendering finished." << std::endl;

	return 0;
}
