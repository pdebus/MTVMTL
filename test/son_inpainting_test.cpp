#include <iostream>


//#define TV_SON_DEBUG
//#define TV_DATA_DEBUG
//#define TV_FUNC_DEBUG_VERBOSE
//#define TVMTL_TVMIN_DEBUG_VERBOSE
//#define TV_VISUAL_DEBUG

#include <mtvmtl/core/algo_traits.hpp>
#include <mtvmtl/core/data.hpp>
#include <mtvmtl/core/functional.hpp>
#include <mtvmtl/core/tvmin.hpp>
#include <mtvmtl/core/visualization.hpp>

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
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;
	typedef Visualization<SO, 3, data_t> visual_t;

	data_t myData = data_t();

	myData.create_nonsmooth_son(30,30);
	myData.output_matval_img("son_img1.csv");
	myData.createRandInpWeights(0.4);
	myData.inpaint_ = true;


	double lam=0.1;
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-10);

	visual_t myVisual(myData);
	
	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SO(N) Cubes Visualization 30x30 Original Picture");
	std::cout << "Rendering finished." << std::endl;
	

	tvmin_t myTVMin(myFunc, myData);

	std::cout << "Interpolate missing pixels to obtain first guess for Newton algorithm..." << std::endl;
	myTVMin.first_guess();
	
	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();

	myVisual.paint_inpainted_pixel(true);
	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SO(N) Cubes Visualization 30x30 Inpainted Picture");
	std::cout << "Rendering finished." << std::endl;

	return 0;
}
