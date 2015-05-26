#ifndef TVTML_DATA_HPP
#define TVTML_DATA_HPP

// system includes
#include <limits>
#include <iostream>
#include <fstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef TV_DATA_DEBUG
    #include <opencv2/highgui/highgui.hpp>
#endif

// video++ includes
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

// own includes 
#include "manifold.hpp"

namespace tvmtl{

// Primary Template
template <typename MANIFOLD, int DIM >
class Data {
};

// Specialization 2D Data
template < typename MANIFOLD >
class Data< MANIFOLD, 2>{

    public:
	static const int img_dim;
	// Manifold typedefs
	typedef typename MANIFOLD::value_type value_type;
	

	// Storage typedefs
	typedef vpp::image2d<value_type> storage_type;
	
	typedef double weights_type;
	typedef vpp::image2d<weights_type> weights_mat;
	
	typedef bool inp_type;
	typedef vpp::image2d<inp_type> inp_mat;

	// Input functions
	void rgb_imread(const char* filename); 
	void findEdgeWeights(); 
	void findInpWeights(const int channel=2);

	inline bool doInpaint() const { return inpaint_; }

	void output_weights(const weights_mat& mat, const char* filename) const;

//    private:
	// Data members
	// TODO Don't forget to initialize with 1px border
	// alignment defaults to 16byte for SSE/SSE2, 32 Byte for AVX
	storage_type img_;
	storage_type noise_img_;
	weights_mat weights_;
	weights_mat edge_weights_;

	bool inpaint_;
	inp_mat inp_; 
};


// Specialization 3D Data
template < typename MANIFOLD >
class Data<MANIFOLD, 3>{

};


/*----- Implementation 2D Data ------*/
template < typename MANIFOLD >
const int Data<MANIFOLD, 2>::img_dim = 2;

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::findEdgeWeights(){
    #ifdef TV_DATA_DEBUG
	std::cout << "Find Edges..." << std::endl;
    #endif
    cv::Mat gray, edge;
    
    { // Local Scope to save memory
	vpp::image2d<vpp::vuchar3> ucharimg(noise_img_.domain());
	
	// Convert double to uchar function
	auto double2uchar = [] (auto& i, const auto& n) {
	    value_type v = n * static_cast<typename MANIFOLD::scalar_type>(std::numeric_limits<unsigned char>::max());
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=static_cast<unsigned char>(v[2]);
	    vu[1]=static_cast<unsigned char>(v[1]);
	    vu[2]=static_cast<unsigned char>(v[0]);
	    i = vu;
	}; 

	vpp::pixel_wise(ucharimg, noise_img_) | double2uchar;
	cv::Mat src = vpp::to_opencv(ucharimg);
	cv::cvtColor(src, gray, CV_BGR2GRAY);
    }

    cv::blur(gray, edge, cv::Size(3,3));
    cv::Canny(edge, edge, 50, 150, 3);
    
    #ifdef TV_DATA_DEBUG
	cv::namedWindow( "Detected Edges", cv::WINDOW_NORMAL ); 
	cv::imshow("Detected Edges", edge);
	cv::waitKey(0);
    #endif

    vpp::image2d<unsigned char> ucharweights = vpp::from_opencv<unsigned char>(edge);
    vpp::pixel_wise(ucharweights, edge_weights_)() | [] (const unsigned char &uw, weights_type& ew) {
	ew = 1.0 - 0.99 * ( static_cast<weights_type>(uw) / static_cast<weights_type>(std::numeric_limits<unsigned char>::max()) );
    };

    #ifdef TV_DATA_DEBUG
	output_weights(edge_weights_, "wedge.csv");
    #endif
}

//TODO: static assert to avoid data that has not exactly 3 channels
// static assert that channel is element {1,2,3}
// make enum RGB out
template < typename MANIFOLD >
void Data<MANIFOLD, 2>::findInpWeights(const int channel){
    #ifdef TV_DATA_DEBUG
	std::cout << "Find Inpainting Area..." << std::endl;
    #endif
    inp_ = inp_mat(noise_img_.domain());
    vpp::pixel_wise(noise_img_, inp_) | [&] (const value_type& img, inp_type& inp) { inp = static_cast<inp_type>(img[channel-1] > 0.95); };
    //vpp::pixel_wise(noise_img_, inp_) | [&] (const value_type& img, inp_type& inp) { inp = static_cast<inp_type>(img[channel-1] > 0.95); };
    #ifdef TV_DATA_DEBUG
	int nr = inp_.nrows();
	int nc = inp_.ncols();

	std::fstream f;
	f.open("inp.csv", std::fstream::out);

	for (int r=0; r<nr; r++){
	    const inp_type* cur = &inp_(r,0);
	    for (int c=0; c<nc; c++){
		f << cur[c];
		if(c != nc-1) f << ",";
	    }
	    f <<  std::endl;
	}
	f.close();
    #endif
}


//TODO: static assert to avoid data that has not exactly 3 channels
template < typename MANIFOLD >
void Data<MANIFOLD, 2>::rgb_imread(const char* filename){
	vpp::image2d<vpp::vuchar3> input_image;
	input_image = vpp::clone(vpp::from_opencv<vpp::vuchar3 >(cv::imread(filename)));
	noise_img_ = storage_type(input_image.domain());
	// Convert Picture of uchar to double 
	    vpp::pixel_wise(input_image, noise_img_) | [] (auto& i, auto& n) {
	    value_type v = value_type::Zero();
	    vpp::vuchar3 vu = i;
	    // TODO: insert manifold scalar type
	    v[0]=static_cast<double>(vu[2]); //opencv saves as BGR
	    v[1]=static_cast<double>(vu[1]);
	    v[2]=static_cast<double>(vu[0]);
	    n = v / static_cast<double>(std::numeric_limits<unsigned char>::max());
	};
    //img_ = vpp::clone(noise_img_, vpp::_border = 1);
    img_ = vpp::clone(noise_img_);
    //Testnoise for Functional Debugging 
    //vpp::pixel_wise(img_) | [] (auto & i) { i*=0.9999; };


    //TODO: Write separate input functions for weights and inpainting matrices
    weights_ = weights_mat(noise_img_.domain());
    vpp::fill(weights_, 1.0);
    edge_weights_ = vpp::clone(weights_);
    vpp::fill(edge_weights_, 1.0);
    inpaint_ = false;
}


template < typename MANIFOLD >
void Data<MANIFOLD, 2>::output_weights(const weights_mat& weights, const char* filename) const{
    int nr = weights.nrows();
    int nc = weights.ncols();

    std::fstream f;
    f.open(filename, std::fstream::out);

    for (int r=0; r<nr; r++){
	const typename Data<MANIFOLD,2>::weights_type* cur = &weights(r,0);
	for (int c=0; c<nc; c++){
	    f << cur[c];
	    if(c != nc-1) f << ",";
	}
	f <<  std::endl;
    }
    f.close();
}

}// end namespace tvmtl

#endif
