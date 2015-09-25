#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>


#define TVMTL_TVMIN_DEBUG
//#define TV_FUNC_DEBUG
//#define TV_FUNC_DEBUG_VERBOSE
//#define TVMTL_TVMIN_DEBUG_VERBOSE
//#define TV_DATA_DEBUG

//#include "../test/imageNd.hh"

#include <mtvmtl/core/algo_traits.hpp>
#include <mtvmtl/core/data.hpp>
#include <mtvmtl/core/functional.hpp>
#include <mtvmtl/core/tvmin.hpp>
#include <mtvmtl/core/visualization.hpp>

#include <vpp/vpp.hh>

using namespace tvmtl;

typedef Manifold< EUCLIDIAN, 1 > mf_t;
typedef Data< mf_t, 3> data_t;	
typedef Functional<FIRSTORDER, ANISO, mf_t, data_t, 3> func_t;
typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP, 3 > tvmin_t;
typedef Visualization<EUCLIDIAN, 1, data_t, 3> visual_t;


int main(int argc, const char *argv[])
{
    if(argc!=4){
	std::cout << "Usage: " << argv[0] << "zDim yDim xDim " << std::endl;
	return 1;
    }

    int nz=atoi(argv[1]);
    int ny=atoi(argv[2]);
    int nx=atoi(argv[3]);
    double lam=0.1;
	    
    data_t myData=data_t();
    myData.create_noisy_gray(nz, ny, nx);
    visual_t myVisual(myData);

    myVisual.saveImage("noisy3dRGB.png");
    std::cout << "Starting OpenGL-Renderer..." << std::endl;
    myVisual.GLInit("Image Cube Renderer ");
    std::cout << "Rendering finished." << std::endl;

    func_t myFunc(lam, myData);
    myFunc.seteps2(1e-16);

    tvmin_t myTVMin(myFunc, myData);
    
    myTVMin.smoothening(5);
    myTVMin.minimize();

    myVisual.saveImage("noisy3dRGB.png");
    std::cout << "Starting OpenGL-Renderer..." << std::endl;
    myVisual.GLInit("Image Cube Renderer ");
    std::cout << "Rendering finished." << std::endl;

    return 0;
}
