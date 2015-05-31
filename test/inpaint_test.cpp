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

typedef Manifold< EUCLIDIAN, 3 > mf_t;
typedef Data< mf_t, 2> data_t;	
typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;


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
	myData.inpaint_=true;
	
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-10);

	tvmin_t myTVMin(myFunc, myData);

	vpp::image2d<vpp::vuchar3> img;
	
	if(myData.doInpaint()){
	    myData.findInpWeights(3);
	    myTVMin.first_guess();
	}
	std::string fname(argv[1]);
	
	DisplayImage("First Guess", myData.img_, img);
	cv::imwrite("firstguess_" + fname, to_opencv(img));

	/*
	std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
	myTVMin.smoothening(10);
	DisplayImage("Smoothened", myData.img_, img);
	cv::imwrite("smoothened_" + fname, to_opencv(img));
	*/

	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();
		

	DisplayImage("Denoised", myData.img_, img);
	cv::imwrite("denoised_" + fname, to_opencv(img));

	return 0;
}
