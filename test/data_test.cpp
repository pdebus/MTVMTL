#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "../core/data.hpp"

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
	typedef typename mf_t::value_type vec3d;

	typedef Data< mf_t, 2> data_t;

	data_t myData=data_t();
	
	myData.rgb_imread(argv[1]);

	cv::namedWindow( "Display window", cv::WINDOW_NORMAL ); 

	// Convert Picture of doubles to uchar
	vpp::image2d<vpp::vuchar3> img(myData.noise_img_.domain());
	vpp::pixel_wise(img, myData.noise_img_) | [] (auto& i, auto& n) {
	    vec3d v = n;
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) n[0];
	    vu[1]=(unsigned char) n[1];
	    vu[2]=(unsigned char) n[2];
	    i = vu;
	};

	cv::imshow( "Display window", vpp::to_opencv(img));
	cv::waitKey(0);
	return 0;
}
