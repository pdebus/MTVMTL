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
	typedef typename data_t::storage_type store_type;

	data_t myData=data_t();
	
	vpp::image2d<vpp::vuchar3> input_image;
	
	input_image = vpp::clone(vpp::from_opencv<vpp::vuchar3 >(cv::imread(argv[1])));
	myData.noise_img_ = store_type(input_image.domain(), vpp::_border = 1);
	// Convert Picture of uchar to double and set one color channel to 0
	    vpp::pixel_wise(input_image, myData.noise_img_) | [] (auto& i, auto& n) {
	    vec3d v = vec3d::Zero();
	    vpp::vuchar3 vu = i;
	    v[0]=(double) vu[0]*0.0;
	    v[1]=(double) vu[1];
	    v[2]=(double) vu[2];
	    n = v;
	};
	//myData.rgb_imread(argv[1]);

	cv::namedWindow( "Display window", cv::WINDOW_NORMAL ); 

	// Convert Picture of double to uchar
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
	//cv::imshow( "Display window", vpp::to_opencv(myData.noise_img_));

	cv::waitKey(0);
	return 0;
}
