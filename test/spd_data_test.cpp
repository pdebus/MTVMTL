#include <iostream>


#define TV_DATA_DEBUG
//#define TV_VISUAL_DEBUG

#include "../core/data.hpp"
#include "../core/visualization.hpp"

int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	typedef Manifold< SPD, 3 > mf_t;

	typedef Data< mf_t, 2> data_t;	

	data_t myData = data_t();

	myData.create_nonsmooth_spd(25,25);
	myData.output_matval_img("spd_img.csv");

	
	typedef Visualization<SPD, 3, data_t> visual_t;

	visual_t myVisual(myData);
	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(N) Ellipsoid Visualization 25x25 ");
	std::cout << "Rendering finished." << std::endl;
	
	return 0;
}
