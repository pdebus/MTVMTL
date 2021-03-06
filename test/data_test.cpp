#include <iostream>
#include <limits>

#include <opencv2/highgui/highgui.hpp>

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

#define TV_DATA_DEBUG
#include <mtvmtl/core/data.hpp>
int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	if (argc != 4){
	    std::cerr << "Usage : " << argv[0] << " csvfile ny nx" << std::endl;
	    return 1;
	}

	typedef Manifold< EUCLIDIAN, 3 > mf_t;
	typedef typename mf_t::value_type vec3d;

	typedef Data< mf_t, 2> data_t;	
	typedef typename data_t::storage_type store_type;

	data_t myData=data_t();
	int nx = atoi(argv[3]);
	int ny = atoi(argv[2]);
	//myData.rgb_imread(argv[1]);
	//myData.findEdgeWeights();
	//myData.findInpWeights(3);
	myData.readMatrixDataFromCSV(argv[1],nx,ny);
	double totalerror = vpp::sum( vpp::pixel_wise(myData.img_, myData.img_) | [](const auto& i, const auto& s) {return mf_t::dist_squared(i,s);} );
	double eucerror  = vpp::sum( vpp::pixel_wise(myData.img_, myData.img_) | [](const auto& i, const auto& s) {return (i-s).cwiseAbs().sum();} );
	
	std::cout << "Geodesic distance error: " << totalerror << std::endl;
	std::cout << "Euclidian distance error: " << eucerror << std::endl;

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
