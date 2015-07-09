#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>

//#define TV_SPD_DEBUG
//#define TV_DATA_DEBUG
#define TV_FUNC_DEBUG
#define TV_FUNC_DEBUG_VERBOSE

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/visualization.hpp"

int main(int argc, const char *argv[])
{
	using namespace tvmtl;
	
	typedef Manifold< SPD, 3 > mf_t;
	typedef Data< mf_t, 2> data_t;	
	typedef Functional<FIRSTORDER, ANISO, mf_t, data_t> func_t;
	typedef Visualization<SPD, 3, data_t> visual_t;

	data_t myData = data_t();
	int ny, nx;
	ny = std::atoi(argv[2]);
	nx = std::atoi(argv[3]);
	myData.readMatrixDataFromCSV(argv[1], ny, nx);


	visual_t myVisual(myData);
	std::stringstream fname;
	std::string nfname;
	fname << "dti" << ny << "x" << ny << ".png";
	nfname = "noisy_" + fname.str();
	myVisual.saveImage(nfname);

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(3) Ellipsoid Visualization");
	std::cout << "Rendering finished." << std::endl;
	
	double lam=0.7;
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-16);
	
	func_t::result_type result = 0.0;
	
	//Functional evaluation
	result = myFunc.evaluateJ();
	//Gradient
	myFunc.evaluateDJ();
	//Hessian
	myFunc.evaluateHJ();

	return 0;
}
