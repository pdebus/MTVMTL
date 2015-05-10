#ifndef TVTML_DATA_HPP
#define TVTML_DATA_HPP

// system includes
#include <limits>
#include <iostream>
#include <fstream>


// video++ includes
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

// Eigen includes

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
	typedef vpp::image2d<short int> inp_mat;

	void rgb_imread(const char* filename); 

	inline bool doInpaint() const { return inpaint_; }

	void output_weights(const weights_mat& mat, const char* filename) const;

//    private:
	// Data members
	// TODO Don't forget to initialize with 1px border
	// alignment defaults to 16byte for SSE/SSE2, 32 Byte for AVX
	storage_type img_;
	storage_type noise_img_;
	weights_mat weights_;

	bool inpaint_;
	inp_mat inp_; 
	weights_mat iweights_;
};


// Specialization 3D Data
template < typename MANIFOLD >
class Data<MANIFOLD, 3>{

};


/*----- Implementation 2D Data ------*/
template < typename MANIFOLD >
const int Data<MANIFOLD, 2>::img_dim = 2;

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
	    // TODO: insert manifold scalar type, replace c-style casts
	    v[0]=(double) vu[2]; //opencv saves as BGR
	    v[1]=(double) vu[1];
	    v[2]=(double) vu[0];
	    n = v / (double) std::numeric_limits<unsigned char>::max();
	};
    //img_ = vpp::clone(noise_img_, vpp::_border = 1);
    img_ = vpp::clone(noise_img_);
    //TODO: Remove test: make noise different to image 
    vpp::pixel_wise(img_) | [] (auto & i) { i*=0.9999; };
    //
    weights_ = weights_mat(noise_img_.domain());
    iweights_ = vpp::clone(weights_);
    vpp::fill(iweights_, 1.0);
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
