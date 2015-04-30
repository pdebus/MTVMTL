#ifndef TVTML_FUNCTIONAL_HPP
#define TVTML_FUNCTIONAL_HPP

// video++ includes
#include <vpp/vpp.hh>

// system includes
#include <cmath>

// own includes 
#include "enumerators.hpp"
#include "manifold.hpp"
#include "data.hpp"

namespace tvmtl{


// Primary Template
// FIXME: 
// - template template parameter necessary here
// - Manifold template parameters necessary or extractable from DATA?
template <enum FUNCTIONAL_ORDER ord, enum FUNCTIONAL_DISC disc, class MANIFOLD, class DATA >
class Functional {
};


template < typename MANIFOLD, class DATA >
class Functional< FIRSTORDER, ISO, MANIFOLD, DATA >{

    public:
	// Manifold typedefs and constants
	static const MANIFOLD_TYPE mf_type = MANIFOLD::MyType;
	typedef typename MANIFOLD::value_type value_type;
	typedef typename MANIFOLD::ref_type ref_type;
	typedef typename MANIFOLD::cref_type cref_type;

	// Data typedef and constants
	static const int img_dim = DATA::img_dim;
	typedef typename DATA::storage_type img_type;
	typedef typename DATA::weights_type weights_type;
	typedef typename DATA::weights_mat weights_mat;
	typedef typename DATA::inp_mat inp_mat;

	// FIXME: Check for box_nbhNd
	// typedef box_nbhNd<value_type, vint<N> >
	typedef vpp::box_nbh2d<value_type,3,3> nbh_type;


	// Functional parameters and return types
	typedef double return_type;
	typedef double param_type;

	//Constructor
	Functional(param_type lambda, DATA dat):
	    lambda_(lambda),
	    data_(dat)
	{eps2_=0.0;}
	
	// Evaluation functions
	return_type evaluateJ();
	return_type evaluateDJ();


    private:
	param_type lambda_;
	param_type eps2_;
	DATA& data_;
};


//--------Implementation FIRSTORDER, ISO-----/

template < typename MANIFOLD, class DATA >
typename Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::return_type Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::evaluateJ(){

    // sum d^2(img, img_noise)
    return_type J1, J2;
    J1 = J2 = 0.0;

    if(data_.doInpaint()){
	auto f = [] (value_type& i, value_type& n, bool inp ) { return MANIFOLD::dist_squared(i,n)*(1-inp); };
	J1 = vpp::sum(vpp::pixel_wise(data_.img_, data_.noise_img_, data_.inp_) | f);
    }
    else{
	auto f = [] (value_type& i, value_type& n) { return MANIFOLD::dist_squared(i,n); };
	J1 = vpp::sum(vpp::pixel_wise(data_.img_, data_.noise_img_) | f);
    }

    // Neighbourhood box
    nbh_type N = nbh_type(data_.img_); 

    weights_mat X,Y;
    X = weights_mat(data_.img_.domain());
    Y = weights_mat(data_.img_.domain());

    // Horizontal Neighbours
    vpp::pixel_wise(X, N)(vpp::_row_backward) | [&] (weights_type& x, auto& nbh) { x = MANIFOLD::dist_squared(nbh(0,0),nbh(1,0)); };
    // Vertical Neighbours
    vpp::pixel_wise(Y, N)(vpp::_col_backward) | [&] (weights_type& y, auto& nbh) { y = MANIFOLD::dist_squared(nbh(0,0),nbh(0,1)); };

    /*
    std::cout << "\nX-Weights:" << std::endl;
    data_.output_weights(X);
    std::cout << "\nY-Weights:" << std::endl;
    data_.output_weights(Y);
    std::cout << std::endl;
    */

    // Compute IRLS Weights
    // TODO:	- Maybe put in different function
    //		- Calculation of weights and inverse weights could also be put in single pixel_wise
    auto g =  [&] (weights_type& iw, weights_type& x, weights_type& y) { return iw / std::sqrt(x+y+eps2_); };
    data_.weights_ = vpp::pixel_wise(data_.iweights_, X, Y) | g ;

    
    //std::cout << "\nIRLS Weights:" << std::endl;
    //data_.output_weights(data_.weights_);
    

    J2 = vpp::sum( vpp::pixel_wise(data_.weights_) | [&] (weights_type& w) {return 1.0/w;} );

    //std::cout << "J1: " << J1 << std::endl;
    //std::cout << "J2: " << J2 << std::endl;

    return 0.5 * J1 + lambda_* J2;
}

}// end namespace tvtml

#endif
