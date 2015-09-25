#include <iostream>


#define TV_DATA_DEBUG
//#define TV_VISUAL_DEBUG

#include <mtvmtl/core/data.hpp>
#include <mtvmtl/core/visualization.hpp>

int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	typedef Manifold< SPD, 3 > mf_t;

	typedef Data< mf_t, 2> data_t;	

	data_t myData = data_t();

	myData.readMatrixDataFromCSV(argv[1], 32, 32);
	myData.output_matval_img("dti_img_test.csv");

	
	typedef Visualization<SPD, 3, data_t> visual_t;

	visual_t myVisual(myData);
	myVisual.saveImage("dti_32x32.png");

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(N) Ellipsoid Visualization 32x32 ");
	std::cout << "Rendering finished." << std::endl;
	
	return 0;
}
