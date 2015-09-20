#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>



using namespace tvmtl;
typedef Manifold< EUCLIDIAN, 1 > eucmf_t;
typedef Data< eucmf_t, 2> bright_t;	
typedef Functional<FIRSTORDER, ANISO, eucmf_t, bright_t> bfunc_t;
typedef TV_Minimizer< IRLS, bfunc_t, eucmf_t, bright_t, OMP > btvmin_t;


int main(int argc, const char *argv[])
{
	Eigen::initParallel();
	
	if (argc < 2){
	    std::cerr << "Usage : " << argv[0] << " image [lambda]" << std::endl;
	    return 1;
	}

	double lam=0.05;

	if(argc==3)
	    lam=atof(argv[2]);
	
	std::string fname(argv[1]);
	
	bright_t myBright=bright_t();

	myBright.rgb_readBrightness(argv[1]);
	vpp::image2d<vpp::vuchar3> img(myBright.img_.domain());
/*	myBright.add_gaussian_noise(0.02);

	// Convert to OpenCV format and show
	vpp::pixel_wise(img, myBright.img_ ) | [] (auto& i, auto& b) {
	    double v =  b[0] * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu.setConstant((unsigned char) v);
	    i = vu;
	};
	cv::namedWindow( "Input Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Picture", vpp::to_opencv(img));
	cv::waitKey(0);

	cv::imwrite("noisy(gray)_" + fname, to_opencv(img));
*/
	bfunc_t bFunc(lam, myBright);
	bFunc.seteps2(1e-10);

	btvmin_t bTVMin(bFunc, myBright);
	
	std::cout << "\n\n--==Brightness PART==--" << std::endl;
	//std::cout << "Smooth picture to obtain initial state for Newton iteration..." << std::endl;
	//bTVMin.smoothening(5);
	
	std::cout << "Start TV minimization..." << std::endl;
	bTVMin.minimize();
		
    	// Convert to OpenCV format and show
	vpp::pixel_wise(img, myBright.img_ ) | [] (auto& i, auto& b) {
	    double v =  b[0] * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu.setConstant((unsigned char) v);
	    i = vu;
	};
	cv::namedWindow( "Input Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Picture", vpp::to_opencv(img));
	cv::waitKey(0);

	cv::imwrite("denoised(gray)_" + fname, to_opencv(img));




	return 0;
}
