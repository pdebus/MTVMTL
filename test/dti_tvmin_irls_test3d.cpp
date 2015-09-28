#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>

//#define TV_SPD_EXP_DEBUG
//#define TV_SPD_LOG_DEBUG

#define TV_DATA_DEBUG
//#define TV_FUNC_DEBUG
//#define TV_FUNC_DEBUG_VERBOSE
//#define TVMTL_TVMIN_DEBUG 
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
	
	typedef Manifold< SPD, 3 > mf_t;
	typedef Data< mf_t, 3> data_t;	
	typedef Functional<FIRSTORDER, ANISO, mf_t, data_t, 3> func_t;
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP, 3 > tvmin_t;
	typedef Visualization<SPD, 3, data_t, 3> visual_t;

	data_t myData = data_t();
	int nz, ny, nx;
	nz = std::atoi(argv[2]);
	ny = std::atoi(argv[3]);
	nx = std::atoi(argv[4]);
	myData.readMatrixDataFromCSV(argv[1], nz, ny, nx);


	visual_t myVisual(myData);
	std::stringstream fname;
	std::string nfname;
	fname << "dti3d" << nz << "x" << ny << "x" << ny << ".png";
	nfname = "noisy_" + fname.str();
	myVisual.saveImage(nfname);

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(3) Ellipsoid Visualization");
	std::cout << "Rendering finished." << std::endl;
	
	double lam=0.7;
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-16);

	tvmin_t myTVMin(myFunc, myData);

	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.smoothening(5);
	myTVMin.minimize();

	std::string dfname = "denoised(irls)_" + fname.str();
	myVisual.saveImage(dfname);

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(3) Ellipsoid Visualization");
	std::cout << "Rendering finished." << std::endl;

	return 0;
}
