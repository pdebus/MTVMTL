#include <iostream>
#include <limits>

#include <opencv2/highgui/highgui.hpp>

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

#define TV_DATA_DEBUG
#include "../core/data.hpp"
int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	if (argc != 2){
	    std::cerr << "Usage : " << argv[0] << " image" << std::endl;
	    return 1;
	}



	typedef Manifold< SPHERE, 3 > spheremf_t;
	typedef Manifold< EUCLIDIAN, 1 > eucmf_t;
	typedef typename spheremf_t::value_type vec3d;
	typedef typename spheremf_t::value_type pxbrightness;

	typedef Data< spheremf_t, 2> chroma_t;	
	typedef Data< eucmf_t, 2> bright_t;	

	chroma_t myChroma=chroma_t();
	bright_t myBright=bright_t();
	
	myChroma.rgb_readChromaticity(argv[1]);
	myBright.rgb_readBrightness(argv[1]);

	// Convert Brightness Picture of double to uchar
	vpp::image2d<vpp::vuchar1> bimg(myBright.img_.domain());
	vpp::pixel_wise(bimg, myBright.img_) | [] (auto& i, auto& n) {
	    double v = n[0] * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar1 vu = vpp::vuchar1::Zero();
	    vu[0]=(unsigned char) v;
	    i = vu;
	};

	cv::namedWindow( "Input Brightness", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Brightness", vpp::to_opencv(bimg));
	cv::waitKey(0);
   
	// Convert Chromaticity Picture of double to uchar
	vpp::image2d<vpp::vuchar3> cimg(myChroma.img_.domain());
	vpp::pixel_wise(cimg, myChroma.img_) | [] (auto& i, auto& n) {
	    vec3d v = n * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) v[2];
	    vu[1]=(unsigned char) v[1];
	    vu[2]=(unsigned char) v[0];
	    i = vu;
	};
	cv::namedWindow( "Input Chromaticty", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Chromaticty", vpp::to_opencv(cimg));
	cv::waitKey(0);


    	// Recombine Brightness and Chromaticity parts
	vpp::image2d<vpp::vuchar3> img(myChroma.img_.domain());
	vpp::pixel_wise(img, myChroma.img_, myBright.img_ ) | [] (auto& i, auto& c, auto& b) {
	    vec3d v = c * b[0] * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) v[2];
	    vu[1]=(unsigned char) v[1];
	    vu[2]=(unsigned char) v[0];
	    i = vu;
	};
	cv::namedWindow( "Input Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Picture", vpp::to_opencv(img));
	cv::waitKey(0);





	return 0;
}
