#include <iostream>


#define TV_DATA_DEBUG
#define TV_VISUAL_CONTROLS_DEBUG

#include "../core/data.hpp"
#include "../core/visualization.hpp"

int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	typedef Manifold< SPD, 3 > mf_t;
	typedef Data< mf_t, 3> data_t;	
	typedef Visualization<SPD, 3, data_t, 3> visual_t;

	data_t myData = data_t();
    
	if(argc > 1){
	    const char* fname = argv[1];
	    int nz=atoi(argv[2]);
	    int ny=atoi(argv[3]);
	    int nx=atoi(argv[4]);
	    myData.readMatrixDataFromCSV(fname, nz, ny, nx);
	}
	else{
	    myData.readMatrixDataFromCSV("dti_image.csv", 8, 16, 8);
	}

	auto setInpaint = [] (const auto& i, auto& inp){
	    inp = ((i - 0.5 * mf_t::value_type::Identity()).norm() < 1e-3);
	};
	myData.inpaint_ = true;
	pixel_wise3d(setInpaint, myData.img_, myData.inp_);

	visual_t myVisual(myData);
	myVisual.saveImage("3Ddti_img.png");

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("SPD(N) Ellipsoid Visualization 8x16x8 ");
	std::cout << "Rendering finished." << std::endl;
	
	return 0;
}
