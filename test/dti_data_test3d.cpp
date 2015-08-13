#include <iostream>


#define TV_DATA_DEBUG
//#define TV_VISUAL_DEBUG

#include "../core/data.hpp"
#include "../core/visualization.hpp"

int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	typedef Manifold< SPD, 3 > mf_t;

	typedef Data< mf_t, 3> data_t;	

	data_t myData = data_t();

	myData.readMatrixDataFromCSV("dti_image.csv", 8, 16, 8);
	
	typedef Visualization<SPD, 3, data_t, 3> visual_t;

	visual_t myVisual(myData);
	myVisual.saveImage("3Ddti_8x16x8.png");

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(N) Ellipsoid Visualization 8x16x8 ");
	std::cout << "Rendering finished." << std::endl;
	
	return 0;
}
