#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>


//#define TVMTL_TVMIN_DEBUG
//#define TV_DATA_DEBUG


#include <mtvmtl/core/algo_traits.hpp>
#include <mtvmtl/core/data.hpp>
#include <mtvmtl/core/functional.hpp>
#include <mtvmtl/core/tvmin.hpp>

#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

using namespace tvmtl;

typedef Manifold< EUCLIDIAN, 3 > mf_t;
typedef Data< mf_t, 2> data_t;	
typedef Functional<FIRSTORDER, ANISO, mf_t, data_t> func_t;
typedef TV_Minimizer< PRPT, func_t, mf_t, data_t, OMP > tvmin_t;


void DisplayImage(const char* wname, const data_t::storage_type& img, vpp::image2d<vpp::vuchar3>& out){
	cv::namedWindow( wname, cv::WINDOW_NORMAL ); 

	// Convert Picture of double to uchar
	vpp::image2d<vpp::vuchar3> vucharimg(img.domain());
	vpp::pixel_wise(vucharimg, img) | [] (auto& i, auto& n) {
	    mf_t::value_type v = n * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) v[2];
	    vu[1]=(unsigned char) v[1];
	    vu[2]=(unsigned char) v[0];
	    i = vu;
	};

	cv::imshow( wname, vpp::to_opencv(vucharimg));

	out =  vucharimg;
	cv::waitKey(0);
}



int main(int argc, const char *argv[])
{

	if (argc < 2){
	    std::cerr << "Usage : " << argv[0] << " image [lambda]" << std::endl;
	    return 1;
	}

	double lam=0.1;

	if(argc==3)
	    lam=atof(argv[2]);

	data_t myData=data_t();
	myData.rgb_imread(argv[1]);

	func_t myFunc(lam, myData);
	myFunc.seteps2(0.0);

	tvmin_t myTVMin(myFunc, myData);
	myTVMin.use_approximate_mean(true);

	vpp::image2d<vpp::vuchar3> img;
	
	std::string fname(argv[1]);
	
	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();
		

	DisplayImage("Denoised", myData.img_, img);
	cv::imwrite("denoised(prpt)_" + fname, to_opencv(img));

	return 0;
}
