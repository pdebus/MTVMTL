#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	if (argc != 2){
	    std::cerr << "Usage : " << argv[0] << " image" << std::endl;
	    return 1;
	}

	typedef Manifold< EUCLIDIAN, 3 > mf_t;
	
	typedef Data< mf_t, 2> data_t;	
	data_t myData=data_t();
	myData.rgb_imread(argv[1]);
	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	func_t myFunc(0.1, myData);
	
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;
	tvmin_t myTVMin(myFunc, myData);
	
	std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
	myTVMin.smoothening(10);
	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();
	
	cv::namedWindow( "Display window", cv::WINDOW_NORMAL ); 

	// Convert Picture of double to uchar
	vpp::image2d<vpp::vuchar3> img(myData.img_.domain());
	vpp::pixel_wise(img, myData.img_) | [] (auto& i, auto& n) {
	    mf_t::value_type v = n * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) v[2];
	    vu[1]=(unsigned char) v[1];
	    vu[2]=(unsigned char) v[0];
	    i = vu;
	};
	
	std::string denoised_fname(argv[1]);
	denoised_fname = "denoised_" + denoised_fname;

	cv::imwrite(denoised_fname, to_opencv(img));
	cv::imshow( "Display window", vpp::to_opencv(img));
	//cv::imshow( "Display window", vpp::to_opencv(myData.noise_img_));

	cv::waitKey(0);

	return 0;
}
