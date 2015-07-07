#include <iostream>
#include <string>
#include <sstream>

//#define TV_SPD_DEBUG
//#define TV_DATA_DEBUG
//#define TV_FUNC_DEBUG_VERBOSE
//#define TVMTL_TVMIN_DEBUG 
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

	int ny, nx;
	if(argc==3){
	    ny = std::atoi(argv[1]);
	    nx = std::atoi(argv[2]);
	}
	else{
	    nx = 30;
	    ny = 30;
	}

	myData.create_nonsmooth_spd(ny ,nx);
	myData.output_matval_img("orignal_spd_img1.csv");
	myData.add_gaussian_noise(0.3);
	myData.output_matval_img("noisy_spd_img1.csv");

	std::stringstream fname;
	std::string nfname;
	fname << "spd" << ny << "x" << ny << ".png";
	nfname = "noisy_" + fname.str();

	visual_t myVisual(myData);
	myVisual.saveImage(nfname);

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(3) Ellipsoid Visualization");
	std::cout << "Rendering finished." << std::endl;
	
	double lam=0.3;
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-10);

	tvmin_t myTVMin(myFunc, myData);

	std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
	myTVMin.smoothening(1);

	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();

	std::string dfname = "denoised_" + fname.str();
	myVisual.saveImage(dfname);

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(3) Ellipsoid Visualization");
	std::cout << "Rendering finished." << std::endl;

	return 0;
}
