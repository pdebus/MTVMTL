#include <iostream>


#define TV_DATA_DEBUG
//#define TV_VISUAL_DEBUG

#include <mtvmtl/core/data.hpp>
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

	data_t myData = data_t();

	myData.create_nonsmooth_son(30,30);
	myData.output_matval_img("son_img.csv");

	typedef Visualization<SO, 3, data_t> visual_t;

	visual_t myVisual(myData);
	myVisual.saveImage("son_30x30.png");
	
	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SO(N) Cubes Visualization 30x30 ");
	std::cout << "Rendering finished." << std::endl;

	return 0;
}
