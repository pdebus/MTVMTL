#ifndef TVTML_DATA_HPP
#define TVTML_DATA_HPP

// system includes

// video++ includes
#include <vpp/vpp.hh>
//#include <vpp/utils/opencv_bridge.hh>

// Eigen includes

// own includes 
#include "manifold.hpp"

namespace tvtml{

// Primary Template
template <typename MANIFOLD, int DIM >
class data {
};

// Specialization 2D Data
template < typename MANIFOLD, 2 >
class data< typename MANIFOLD>{

    public:
	static const int img_dim = 2;
	// Manifold typedefs
	typedef typename MANIFOLD::value_type value_type;
	

	// Storage typedefs
	typedef vpp::image2d<value_type> storage_type;
	typedef vpp::image2d<double> weights_mat;
	typedef vpp::image2d<short int> inp_mat;

	inline void imread(); 

	inline bool doInpaint() const { return inpaint_; }

//    private:
	// Data members
	// FIXME Don't forget to initialize with 1px border
	// alignment defaults to 16byte for SSE
	storage_type img_;
	storage_type noise_img_;
	weights_mat weights_;

	bool inpaint_;
	inp_type inp_; // should be inverted for internal use 
};


// Specialization 3D Data
template < typename MANIFOLD, 3 >
class data< typename MANIFOLD>{

};

}// end namespace tvtml

#endif
