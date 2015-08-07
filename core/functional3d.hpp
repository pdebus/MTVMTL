#ifndef TVTML_FUNCTIONAL3D_HPP
#define TVTML_FUNCTIONAL3D_HPP

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Sparse>

// video++ includes
#include <vpp/vpp.hh>

// system includes
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

// own includes 
#include "enumerators.hpp"
#include "manifold.hpp"
#include "data.hpp"

namespace tvmtl{


template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA>
class Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3>{

    public:
	// Manifold typedefs and constants
	static const MANIFOLD_TYPE mf_type = MANIFOLD::MyType;
	static const int value_dim = MANIFOLD::value_dim; 
	static const int manifold_dim = MANIFOLD::manifold_dim; 
	typedef typename MANIFOLD::scalar_type scalar_type;
	typedef typename MANIFOLD::value_type value_type;
	typedef typename MANIFOLD::ref_type ref_type;
	typedef typename MANIFOLD::cref_type cref_type;
	typedef typename MANIFOLD::deriv1_type deriv1_type;
	typedef typename MANIFOLD::deriv2_type deriv2_type;
	typedef typename MANIFOLD::restricted_deriv2_type restricted_deriv2_type;
	typedef typename MANIFOLD::tm_base_type tm_base_type;

	// Data typedef and constants
	static const int img_dim = DATA::img_dim;
	typedef typename DATA::storage_type img_type;
	typedef typename DATA::weights_type weights_type;
	typedef typename DATA::weights_mat weights_mat;
	typedef typename DATA::inp_mat inp_mat;

	typedef vpp::box_nbh2d<value_type,3,3> nbh_type;

	// Functional parameters and return types
	static const FUNCTIONAL_DISC disc_type;
	typedef double param_type;
	typedef double result_type;
	
	// Tangent space transformation matrix types
	typedef vpp::imageNd<tm_base_type, img_dim> tm_base_mat_type; 
	
	// Gradient and Hessian types
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> gradient_type;
	typedef vpp::imageNd<deriv2_type, img_dim> hessian_type;
	typedef Eigen::SparseMatrix<scalar_type> sparse_hessian_type;

	//Constructor
	Functional(param_type lambda, DATA& dat):
	    lambda_(lambda),
	    data_(dat)
	{
	    eps2_=1e-10;
	   static_assert(img_dim == 3, "Dimension of data and functional must match!");
	}
	
	void updateWeights();

	void updateTMBase();
	
	
	// Evaluation functions
	result_type evaluateJ();
	void  evaluateDJ();
	void  evaluateHJ();
	
	template <class IMG>
	void output_img(const IMG& img, const char* filename) const;
	template <class IMG>
	void output_matval_img(const IMG& img, const char* filename) const;

	// Getter and Setter 
	inline param_type getlambda() const { return lambda_; }
	inline void setlambda(param_type lam) { lambda_=lam; }
	inline param_type geteps2() const { return eps2_; }
	inline void seteps2(param_type eps) { eps2_=eps; }

	inline const weights_mat& getweightsX() const { return weightsX_; }
	inline const weights_mat& getweightsY() const { return weightsY_; }
	inline const weights_mat& getweightsZ() const { return weightsZ_; }

	inline const gradient_type& getDJ() const { return DJ_; }
	inline const sparse_hessian_type& getHJ() const { return HJ_; }
	inline const tm_base_mat_type& getT() const { return T_; }

    private:
	DATA& data_;

	param_type lambda_, eps2_;
	weights_mat weightsX_, weightsY_, weightsZ_;

	tm_base_mat_type T_;
	gradient_type DJ_;
	sparse_hessian_type HJ_;
};


//--------Implementation FIRSTORDER-----/

template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
const FUNCTIONAL_DISC Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3>::disc_type = disc;

template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA>
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3>::updateWeights(){

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t Update Weights..." << std::endl;
    #endif

    // Neighbourhood box
    int ns = data_.img_.nslices();  // z
    int nr = data_.img_.nrows();    // y
    int nc = data_.img_.ncols();    // x
    
    // Subimage boxes
    vpp::box3d without_last_x(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 1, nc - 2)); // subdomain without last xslice
    vpp::box3d without_last_y(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 2, nc - 1)); // subdomain without last yslice
    vpp::box3d without_last_z(vpp::vint3(0,0,0), vpp::vint3(ns - 2, nr - 1, nc - 1)); // subdomain without last zslice
    vpp::box3d without_first_x(vpp::vint3(0,0,1), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first xlice
    vpp::box3d without_first_y(vpp::vint3(0,1,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first yslice
    vpp::box3d without_first_z(vpp::vint3(1,0,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first zslice

    weightsX_ = weights_mat(data_.img_.domain());
    weightsY_ = weights_mat(data_.img_.domain());
    weightsZ_ = weights_mat(data_.img_.domain());

    auto calc_dist = [&] (weights_type& w, const value_type i, const value_type n) {
	i = MANIFOLD::dist_squared(i, n);
    };

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...X neighbours " << std::endl;
    #endif
    
    // X Neighbours
    vpp::fill(weightsX_, 0.0);
    vpp::pixel_wise(weightsX_ | without_last_x, data_.img_ | without_last_x, data_.img_ | without_first_x )(/*vpp::_no_threads*/) | calc_dist; 

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Y neighbours" << std::endl;
    #endif

    // Y Neighbours
    vpp::fill(weightsY_, 0.0);
    vpp::pixel_wise(weightsY_ | without_last_y, data_.img_ | without_last_y, data_.img_ | without_first_y )(/*vpp::_no_threads*/) | calc_dist;

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Reweighting" << std::endl;
    #endif

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Z neighbours" << std::endl;
    #endif

    // Z Neighbours
    vpp::fill(weightsZ_, 0.0);
    vpp::pixel_wise(weightsZ_ | without_last_z, data_.img_ | without_last_z, data_.img_ | without_first_z )(/*vpp::_no_threads*/) | calc_dist;

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Reweighting" << std::endl;
    #endif

    if(disc==ISO){
	auto g =  [&] (const weights_type& ew, weights_type& x, weights_type& y, weights_type& z) { x = ew / std::sqrt(x + y + z + eps2_); y = x; z = x; };
	vpp::pixel_wise(data_.edge_weights_, weightsX_, weightsY_, weightsZ_) | g ;
    }
    else{
	auto g =  [&] (const weights_type& ew, weights_type& w) { w = ew / std::sqrt(w+eps2_); };
	vpp::pixel_wise(data_.edge_weights_, weightsX_) | g ;
	vpp::pixel_wise(data_.edge_weights_, weightsY_) | g ;
	vpp::pixel_wise(data_.edge_weights_, weightsZ_) | g ;
    }
}

// Update the Tangent space ONB
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::updateTMBase(){
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tUpdate tangent space basis..." << std::endl;
    #endif


   tm_base_mat_type T(data_.img_.domain());
   vpp::pixel_wise(T, data_.img_)(/*vpp::_no_threads*/) | [&] (tm_base_type& t, const value_type& i) { MANIFOLD::tangent_plane_base(i,t); };
   T_=T;
    
}


// Evaluation of J
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
typename Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::result_type Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::evaluateJ(){

    // sum d^2(img, img_noise)
    result_type J1, J2;
    J1 = J2 = 0.0;
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tFunctional evaluation..." << std::endl;
	std::cout << "\t\t...Fidelity part" << std::endl;
    #endif


    if(data_.doInpaint()){
	auto f = [] (const value_type& i, const value_type& n, const bool inp ) { return MANIFOLD::dist_squared(i,n)*(1-inp); };
	J1 = vpp::sum(vpp::pixel_wise(data_.img_, data_.noise_img_, data_.inp_) | f);
    }
    else{
	auto f = [] (const value_type& i, const value_type& n) { return MANIFOLD::dist_squared(i,n); };
	J1 = vpp::sum(vpp::pixel_wise(data_.img_, data_.noise_img_)(/*vpp::_no_threads*/)| f);
    }

	updateWeights();

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...TV part." << std::endl;
    #endif

    if(disc==ISO)
	J2 = vpp::sum( vpp::pixel_wise(weightsX_) | [&] (const weights_type& w) {return 1.0/w;} );
    else
	J2 = vpp::sum( vpp::pixel_wise(weightsX_, weightsY_, weightsZ_) | [&] (const weights_type& wx, const weights_type& wy, const weights_type& wz) {return 1.0 / wx + 1.0 / wy+ 1.0 / wz;} );
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "J1: " << J1 << std::endl;
	std::cout << "J2: " << J2 << std::endl;
    #endif

    return 0.5 * J1 + lambda_* J2;
}

// Evaluation of J'
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::evaluateDJ(){
    //TODO
}

// Evaluation of Hessian J
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::evaluateHJ(){
    //TODO
}


template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
template < class IMG >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::output_img(const IMG& img, const char* filename) const{
    //TODO
}

template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
template < class IMG >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::output_matval_img(const IMG& img, const char* filename) const{
    //TODO
}

}// end namespace tvtml

#endif
