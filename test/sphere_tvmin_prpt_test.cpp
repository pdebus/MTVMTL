#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>

#include <mtvmtl/core/algo_traits.hpp>
#include <mtvmtl/core/data.hpp>
#include <mtvmtl/core/functional.hpp>
#include <mtvmtl/core/tvmin.hpp>

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>



using namespace tvmtl;
typedef Manifold< SPHERE, 3 > spheremf_t;
typedef Manifold< EUCLIDIAN, 1 > eucmf_t;

typedef Data< spheremf_t, 2> chroma_t;	
typedef Data< eucmf_t, 2> bright_t;	

typedef Functional<FIRSTORDER, ANISO, spheremf_t, chroma_t> cfunc_t;
typedef Functional<FIRSTORDER, ANISO, eucmf_t, bright_t> bfunc_t;

typedef TV_Minimizer< PRPT, cfunc_t, spheremf_t, chroma_t, OMP > ctvmin_t;
typedef TV_Minimizer< PRPT, bfunc_t, eucmf_t, bright_t, OMP > btvmin_t;


int main(int argc, const char *argv[])
{
	Eigen::initParallel();
	
	if (argc < 2){
	    std::cerr << "Usage : " << argv[0] << " image [lambda]" << std::endl;
	    return 1;
	}

	double lam=0.1;

	if(argc==3)
	    lam=atof(argv[2]);
	
	std::string fname(argv[1]);
	
	chroma_t myChroma=chroma_t();
	bright_t myBright=bright_t();
	
	myChroma.rgb_readChromaticity(argv[1]);
	myBright.rgb_readBrightness(argv[1]);

	cfunc_t cFunc(lam, myChroma);
	cFunc.seteps2(0.0);

	bfunc_t bFunc(lam, myBright);
	bFunc.seteps2(0.0);

	ctvmin_t cTVMin(cFunc, myChroma);
	btvmin_t bTVMin(bFunc, myBright);
	
	std::cout << "\n\n--==Brightness PART==--" << std::endl;
	
	std::cout << "Start TV minimization..." << std::endl;
	bTVMin.minimize();
		
	std::cout << "\n\n--==CHROMATICITY PART==--" << std::endl;
	
	std::cout << "Start TV minimization..." << std::endl;
	cTVMin.minimize();
		
	

    	// Recombine Brightness and Chromaticity parts
	vpp::image2d<vpp::vuchar3> img(myChroma.img_.domain());
	vpp::pixel_wise(img, myChroma.img_, myBright.img_ ) | [] (auto& i, auto& c, auto& b) {
	    vpp::vdouble3 v = c * b[0] * std::sqrt(3)  * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) v[2];
	    vu[1]=(unsigned char) v[1];
	    vu[2]=(unsigned char) v[0];
	    i = vu;
	};
	cv::namedWindow( "Input Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Picture", vpp::to_opencv(img));
	cv::waitKey(0);

	cv::imwrite("denoised(PRPT-CBR)_" + fname, to_opencv(img));

	return 0;
}
