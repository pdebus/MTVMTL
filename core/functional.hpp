#ifndef TVTML_FUNCTIONAL_HPP
#define TVTML_FUNCTIONAL_HPP

// video++ includes
#include <vpp/vpp.hh>

// system includes

// own includes 
#include "manifold.hpp"
#include "data.hpp"

namespace tvtml{

// Primary Template
// FIXME: 
// - template template parameter necessary here
// - Manifold template parameters necessary or extractable from DATA?
template <enum ORDER ord, enum DISC disc, class MANIFOLD, template <class> class DATA >
class functional {
};


template < typename MANIFOLD, class DATA >
class functional< FIRSTORDER, ISO >{

    public:
	// Manifold typedefs and constants
	static const MANIFOLD_TYPE mf_type = MANIFOLD::MyType;
	typedef typename MANIFOLD::value_type value_type;
	typedef typename MANIFOLD::ref_type ref_type;
	typedef typename MANIFOLD::cref_type cref_type;

	// Data typedef and constants
	static const img_dim = DATA::img_dim;
	typedef typename DATA::storage_type img_type;
	typedef typename DATA::weights_mat weights_mat;
	typedef typename DATA::inp_mat inp_mat;

	// FIXME: Check for box_nbhNd
	// typedef box_nbhNd<value_type, vint<N> >
	typedef box_nbh2d<value_type,3,3> nbh_type;


	// Functional parameters and return types
	typedef double return_type;
	typedef double param_type;

	//Constructor
	functional(param_type lamba, Data dat):
	    lambda_(lambda),
	    data_(dat);
	
	// Evaluation functions
	return_type evaluateJ() const;
	return_type evaluateDJ() const;

    private:
	param_type lambda_;
	param_type eps2_;
	DATA data_;
};


//--------Implementation FIRSTORDER, ISO-----/

template < typename MANIFOLD, class DATA >
return_type functional< FIRSTORDER, ISO >::evaluateJ(){

    // sum d^2(img, img_noise)
    return_type J1, J2;

    if(data_.inpaint())
	J1 = vpp::sum(data_.img_ - data_.noise_img_);
    else
	J1 = vpp::sum((data_.img_ - data_.noise_img_)*data_.inp_);
   
    // Neighbourhood box
    nbh_type N = nbh_type(data_.img); 
    // Horizontal Neighbours
    weights_mat X = vpp::pixel_wise(N)(_Row_backward) | [&] (auto& nbh) { return MANIFOLD::dist_squared(nbh(0,0),nbh(1,0)); }
    // Vertical Neighbours
    weights_mat Y = vpp::pixel_wise(N)(_Col_backward) | [&] (auto& nbh) { return MANIFOLD::dist_squared(nbh(0,0),nbh(0,1)); }

    data_.weights_  = vpp::pixel_wise(X, Y) | [&] (weights_type& x, weights_type& y) { return 1.0/std::sqrt(x+y+eps2_); }
    J2 = vpp::sum(data_.weights_);

    return 0.5 * J1 + lambda_* J2;
}





}// end namespace tvtml

#endif
