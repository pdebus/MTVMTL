#include <iostream>
#include <string>
#include <cmath>

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

typedef Functional<FIRSTORDER, ISO, spheremf_t, chroma_t> cfunc_t;
typedef Functional<FIRSTORDER, ISO, eucmf_t, bright_t> bfunc_t;

typedef TV_Minimizer< IRLS, cfunc_t, spheremf_t, chroma_t, OMP > ctvmin_t;
typedef TV_Minimizer< IRLS, bfunc_t, eucmf_t, bright_t, OMP > btvmin_t;


void removeColor(chroma_t& C,const bright_t& B){
    vpp::pixel_wise(C.img_, B.img_, C.inp_) | [&] (auto& c, const auto& b, const bool& i){	
	if(i){ 
	    c.setConstant(b[0]);

	    if(b[0]!=0)	c.normalize();
	    else c.setConstant(1.0/256.0);
	}
	if(!std::isfinite(c(0))){
	    std::cout << "NaN in RemoveColor" << std::endl;
	    std::cout << b << std::endl;
	}
    };
}

void DisplayImage(const char* wname, const chroma_t& C){
	cv::namedWindow( wname, cv::WINDOW_NORMAL ); 

	// Convert Picture of double to uchar
	vpp::image2d<vpp::vuchar3> vucharimg(C.img_.domain());
	vpp::pixel_wise(vucharimg, C.img_) | [] (auto& i, auto& n) {
	    spheremf_t::value_type v = n * (double) std::numeric_limits<unsigned char>::max();
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) v[2];
	    vu[1]=(unsigned char) v[1];
	    vu[2]=(unsigned char) v[0];
	    i = vu;
	};

	cv::imshow( wname, vpp::to_opencv(vucharimg));
	cv::waitKey(0);
}

void recombineAndShow(const chroma_t& C, const bright_t B, std::string fname, std::string wname){
	
	vpp::image2d<vpp::vuchar3> img(C.img_.domain());
	vpp::pixel_wise(img, C.img_, B.img_ ) | [] (auto& i, const auto& c, const auto& b) {
	    vpp::vdouble3 v = c * b[0] * std::sqrt(3);
	    
	    double max = v.maxCoeff();
	    if(max > 1.0) v /= max;
	    
	    v *= (double) std::numeric_limits<unsigned char>::max();
	    
	    
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=(unsigned char) v[2];
	    vu[1]=(unsigned char) v[1];
	    vu[2]=(unsigned char) v[0];
	    i = vu;
	};
	cv::namedWindow( wname, cv::WINDOW_NORMAL ); 
	cv::imshow( wname, vpp::to_opencv(img));
	cv::waitKey(0);

	cv::imwrite(fname, to_opencv(img));

}

int main(int argc, const char *argv[])
{
	Eigen::initParallel();
	
	if (argc < 3){
	    std::cerr << "Usage : " << argv[0] << " image [lambda] [threshold]" << std::endl;
	    return 1;
	}

	double lam=0.01;
	double threshold=0.01;

	if(argc==4){
	    lam=atof(argv[2]);
	    threshold=atof(argv[3]);
	}

	std::string fname(argv[1]);
	
	chroma_t myChroma=chroma_t();
	bright_t myBright=bright_t();
	
	myBright.rgb_readBrightness(argv[1]);
	myBright.findEdgeWeights();
	
	myChroma.rgb_readChromaticity(argv[1]);
	myChroma.inpaint_=true;
	myChroma.setEdgeWeights(myBright.edge_weights_);
	myChroma.createRandInpWeights(threshold);
	removeColor(myChroma, myBright);
	
	// Recombine Brightness and Chromaticity parts to view Picture with colors removed
	recombineAndShow(myChroma, myBright, "colorless_"+fname, "Colors removed Picture");

	cfunc_t cFunc(lam, myChroma);
	cFunc.seteps2(1e-10);

	ctvmin_t cTVMin(cFunc, myChroma);
	cTVMin.first_guess();
	recombineAndShow(myChroma, myBright, "recolored_fg_"+fname, "Recolor First Guess");
	DisplayImage("Chromaticity First Guess", myChroma);
	
	std::cout << "\n\n--==CHROMATICITY PART==--" << std::endl;

	//std::cout << "Smooth picture to obtain initial state for Newton iteration..." << std::endl;
	//cTVMin.smoothening(10);
	
	std::cout << "Start TV minimization..." << std::endl;
	cTVMin.minimize();
	

    	// Recombine Brightness and Chromaticity parts of recolored Picture
	recombineAndShow(myChroma, myBright, "recolored_"+fname, "Recolored Picture");



	return 0;
}
