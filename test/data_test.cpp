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

	typedef Manifold< EUCLIDIAN, 3 > mf_t;
	typedef typename mf_t::value_type vec3d;

	typedef Data< mf_t, 2> data_t;	
	typedef typename data_t::storage_type store_type;

	data_t myData=data_t();
	
	myData.rgb_imread(argv[1]);
	myData.findEdgeWeights();
	myData.findInpWeights(2);
	cv::namedWindow( "Input Picture", cv::WINDOW_NORMAL ); 

	// Convert Picture of double to uchar
	vpp::image2d<vpp::vuchar3> img(myData.img_.domain());
	vpp::pixel_wise(img, myData.img_) | [] (auto& i, auto& n) {
	    vec3d v = n * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) v[2];
	    vu[1]=(unsigned char) v[1];
	    vu[2]=(unsigned char) v[0];
	    i = vu;
	};
	
	cv::imshow( "Input Picture", vpp::to_opencv(img));
	//cv::imshow( "Display window", vpp::to_opencv(myData.noise_img_));

	cv::waitKey(0);
	return 0;
}
