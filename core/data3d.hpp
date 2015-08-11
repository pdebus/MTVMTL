#ifndef TVTML_DATA3D_HPP
#define TVTML_DATA3D_HPP

// system includes
#include <cassert>
#include <limits>
#include <cmath>
#include <random>

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

//Eigen includes
#include <Eigen/Core>
#include <Eigen/Geometry>

// video++ includes
#include <vpp/vpp.hh>

// own includes 
#include "data3d_utils.hpp"
#include "manifold.hpp"

namespace tvmtl{

// Specialization 3D Data
template < typename MANIFOLD >
class Data<MANIFOLD, 3>{
    
    public:
	static const int img_dim;

	// Manifold typedefs
	typedef typename MANIFOLD::value_type value_type;
	typedef typename MANIFOLD::scalar_type scalar_type;
	
	// Storage typedefs
	typedef vpp::image3d<value_type> storage_type;
	
	typedef double weights_type;
	typedef vpp::image3d<weights_type> weights_mat;
	
	typedef bool inp_type;
	typedef vpp::image3d<inp_type> inp_mat;

	inline bool doInpaint() const { return inpaint_; }
	    
	// Data Init functions
	inline void initEdgeweights();
	inline void initInp();
	
	void create_noisy_gray(const int nz, const int ny, const int nx, double color=0.5, double stdev=0.1);
	void create_noisy_rgb(const int nz, const int ny, const int nx, int color=1, double stdev=0.1);

	void setEdgeWeights(const weights_mat&);

//  private:
	storage_type img_;
	storage_type noise_img_;
	weights_mat edge_weights_;

	bool inpaint_;
	inp_mat inp_; 
};


/*----- Implementation 3D Data ------*/
template < typename MANIFOLD >
const int Data<MANIFOLD, 3>::img_dim = 3;
template < typename MANIFOLD >
void Data<MANIFOLD, 3>::setEdgeWeights(const weights_mat& w){
    clone3d(w, edge_weights_);
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::initInp(){
    inp_ = inp_mat(noise_img_.domain());
    fill3d(inp_, false);
    inpaint_ = false;
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::initEdgeweights(){
    edge_weights_ = weights_mat(noise_img_.domain());
    fill3d(edge_weights_, 1.0);
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::create_noisy_gray(const int nz, const int ny, const int nx, double color, double stdev){
    static_assert(MANIFOLD::value_dim == 1, "Method is only callable for Manifolds with embedding dimension 1");
    noise_img_ = storage_type(nz, ny, nx);
    img_ = storage_type(noise_img_.domain());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<typename MANIFOLD::scalar_type> rand(0.0, stdev);
    auto insert = [&] (value_type& i) { i.setConstant(color + rand(gen)); };

    pixel_wise3d(insert, noise_img_);
    clone3d(noise_img_, img_);
    initInp();
    initEdgeweights();
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::create_noisy_rgb(const int nz, const int ny, const int nx, int color, double stdev){
    static_assert(MANIFOLD::value_dim == 3, "Method is only callable for Manifolds with embedding dimension 3");
    noise_img_ = storage_type(nz, ny, nx);
    img_ = storage_type(noise_img_.domain());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<typename MANIFOLD::scalar_type> rand(0.0, stdev);
    value_type v(0.7 + rand(gen), 0.3 + rand(gen), 0.3 + rand(gen));
    auto insert = [&] (value_type& i) { i = v;  };

    pixel_wise3d(insert, noise_img_);
    clone3d(noise_img_, img_);
    initInp();
    initEdgeweights();
}
}// end namespace tvmtl

#endif