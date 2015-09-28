#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>


//#define TVMTL_TVMIN_DEBUG
//#define TVMTL_TVMIN_DEBUG_VERBOSE
//#define TV_DATA_DEBUG


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
    if(argc<5){
	std::cout << "Usage: " << argv[0] << " filename zDim yDim xDim [lam]" << std::endl;
	return 1;
    }

    std::string fname(argv[1]);
    int nz=atoi(argv[2]);
    int ny=atoi(argv[3]);
    int nx=atoi(argv[4]);
    double lam=0.1;

    if(argc==6)
	lam = atof(argv[5]);

    data_t myData=data_t();
    myData.readRawVolumeData(fname, nz, ny, nx);
    myData.add_gaussian_noise(0.1);
    visual_t myVisual(myData);

    std::size_t pos = fname.find(".csv");
    std::string noisename = "noisy_" + fname.substr(0,pos) + ".jpg";
    
    myVisual.saveImage(noisename.c_str());
    std::cout << "Starting OpenGL-Renderer..." << std::endl;
    myVisual.GLInit("Image Cube Renderer ");
    std::cout << "Rendering finished." << std::endl;

    func_t myFunc(lam, myData);
    myFunc.seteps2(1e-16);

    tvmin_t myTVMin(myFunc, myData);
   
    //myTVMin.smoothening(5);
    //myTVMin.setMax_irls_steps(4);
    myTVMin.minimize();

    std::string dnoisename = "denoised(IRLS)_" + fname.substr(0,pos) + ".jpg";
    myVisual.saveImage(dnoisename.c_str());
    std::cout << "Starting OpenGL-Renderer..." << std::endl;
    myVisual.GLInit("Image Cube Renderer ");
    std::cout << "Rendering finished." << std::endl;

    return 0;
}
