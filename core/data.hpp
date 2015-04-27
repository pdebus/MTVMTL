#ifndef TVTML_DATA_HPP
#define TVTML_DATA_HPP

// system includes

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
	typedef vpp::image2d<double> weights_mat;
	typedef vpp::image2d<short int> inp_mat;

	inline void rgb_imread(const char* filename); 

	inline bool doInpaint() const { return inpaint_; }

//    private:
	// Data members
	// TODO Don't forget to initialize with 1px border
	// alignment defaults to 16byte for SSE/SSE2, 32 Byte for AVX
	storage_type img_;
	storage_type noise_img_;
	weights_mat weights_;

	bool inpaint_;
	inp_mat inp_; // should be inverted for internal use 
};


// Specialization 3D Data
template < typename MANIFOLD >
class Data<MANIFOLD, 3>{

};


/*----- Implementation 2D Data ------*/
template < typename MANIFOLD >
const int Data<MANIFOLD, 2>::img_dim = 2;


template < typename MANIFOLD >
inline void Data<MANIFOLD, 2>::rgb_imread(const char* filename){
    noise_img_ = vpp::clone(vpp::from_opencv<value_type >(cv::imread(filename)), vpp::_border = 1);
    img_ = storage_type(noise_img_.domain());
    weights_ = weights_mat(noise_img_.domain());
}


}// end namespace tvmtl

#endif
