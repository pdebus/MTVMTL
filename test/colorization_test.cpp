#include <iostream>
#include <string>
#include <cmath>

#include <opencv2/highgui/highgui.hpp>

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"

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


void removeColor(chroma_t& C, const bright_t& B){
    vpp::pixel_wise(C.noise_img_, B.noise_img_, C.inp_) | [&] (auto& c, const auto& b, const bool& i){	
	if(i){ 
	    c.setConstant(b[0]);
	    if(b[0]!=0)	c.normalize();
	}
	if(!std::isfinite(c(0))){
	    std::cout << "NaN in RemoveColor" << std::endl;
	    std::cout << b << std::endl;
	}
    };
    C.img_ = vpp::clone(C.noise_img_);
}

void recombineAndShow(const chroma_t& C, const bright_t B, std::string fname, std::string wname){
	
	vpp::image2d<vpp::vuchar3> img(C.img_.domain());
	vpp::pixel_wise(img, C.img_, B.img_ ) | [] (auto& i, const auto& c, const auto& b) {
	    vpp::vdouble3 v = c * b[0] * std::sqrt(3) * (double) std::numeric_limits<unsigned char>::max();
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
	
	if (argc < 2){
	    std::cerr << "Usage : " << argv[0] << " image [lambda]" << std::endl;
	    return 1;
	}

	double lam=0.01;

	if(argc==3)
	    lam=atof(argv[2]);
	
	std::string fname(argv[1]);
	
	chroma_t myChroma=chroma_t();
	bright_t myBright=bright_t();
	
	myBright.rgb_readBrightness(argv[1]);
	myBright.findEdgeWeights();
	
	myChroma.rgb_readChromaticity(argv[1]);
	myChroma.inpaint_=true;
	myChroma.setEdgeWeights(myBright.edge_weights_);
	myChroma.createRandInpWeights(0.01);
	removeColor(myChroma, myBright);
	
	// Recombine Brightness and Chromaticity parts to view Picture with colors removed
	recombineAndShow(myChroma, myBright, "colorless_"+fname, "Colors removed Picture");

	cfunc_t cFunc(lam, myChroma);
	cFunc.seteps2(1e-10);

	ctvmin_t cTVMin(cFunc, myChroma);
	cTVMin.first_guess();
	recombineAndShow(myChroma, myBright, "recolored_fg_"+fname, "Recolor First Guess");
	
	std::cout << "\n\n--==CHROMATICITY PART==--" << std::endl;

	//std::cout << "Smooth picture to obtain initial state for Newton iteration..." << std::endl;
	//cTVMin.smoothening(10);
	
	std::cout << "Start TV minimization..." << std::endl;
	cTVMin.minimize();
	

    	// Recombine Brightness and Chromaticity parts of recolored Picture
	recombineAndShow(myChroma, myBright, "recolored_"+fname, "Recolored Picture");



	return 0;
}
