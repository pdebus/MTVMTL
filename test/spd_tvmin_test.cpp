#include <iostream>


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
	
	typedef Manifold< SPD, 3 > mf_t;
	typedef Data< mf_t, 2> data_t;	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;
	typedef Visualization<SPD, 3, data_t> visual_t;

	data_t myData = data_t();

	myData.create_nonsmooth_spd(30,30);
	myData.output_matval_img("spd_img1.csv");

	double lam=0.3;
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-10);

	tvmin_t myTVMin(myFunc, myData);

	std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
	myTVMin.smoothening(1);

	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();


	visual_t myVisual(myData);
	myVisual.saveImage("denoised_spd30x30.png");

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(3) Ellipsoid Visualization 30x30 ");
	std::cout << "Rendering finished." << std::endl;

	return 0;
}
