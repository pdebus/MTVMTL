#include <iostream>


#define TV_DATA_DEBUG
#include "../core/data.hpp"
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

	typedef Visualization<SO,3> visual_t;

	visual_t myVisual(myData);
	myVisual.GLInit("SO(N) Cubes Visualization 30x30 ");

	return 0;
}
