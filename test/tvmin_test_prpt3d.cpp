#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>


//#define TVMTL_TVMIN_DEBUG
//#define TVMTL_TVMIN_DEBUG_VERBOSE
//#define TV_DATA_DEBUG


#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"
#include "../core/visualization.hpp"

#include <vpp/vpp.hh>

using namespace tvmtl;

typedef Manifold< EUCLIDIAN, 1 > mf_t;
typedef Data< mf_t, 3> data_t;	
typedef Functional<FIRSTORDER, ANISO, mf_t, data_t, 3> func_t;
typedef TV_Minimizer< PRPT, func_t, mf_t, data_t, OMP, 3 > tvmin_t;
typedef Visualization<EUCLIDIAN, 1, data_t, 3> visual_t;


int main(int argc, const char *argv[])
{
    if(argc!=5){
	std::cout << "Usage: " << argv[0] << " filename zDim yDim xDim " << std::endl;
	return 1;
    }

    const char* fname = argv[1];
    int nz=atoi(argv[2]);
    int ny=atoi(argv[3]);
    int nx=atoi(argv[4]);
    double lam=0.1;
	    
    data_t myData=data_t();
    myData.readRawVolumeData(fname, nz, ny, nx);
    myData.add_gaussian_noise(0.1);
    visual_t myVisual(myData);

    myVisual.saveImage("noisyVolumeImg.png");
    std::cout << "Starting OpenGL-Renderer..." << std::endl;
    myVisual.GLInit("Image Cube Renderer ");
    std::cout << "Rendering finished." << std::endl;

    func_t myFunc(lam, myData);
    myFunc.seteps2(0.0);

    tvmin_t myTVMin(myFunc, myData);
    myTVMin.use_approximate_mean(false);
    
    myTVMin.minimize();

    myVisual.saveImage("denoisedVolumeImg.png");
    std::cout << "Starting OpenGL-Renderer..." << std::endl;
    myVisual.GLInit("Image Cube Renderer ");
    std::cout << "Rendering finished." << std::endl;

    return 0;
}
