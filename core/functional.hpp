#ifndef TVTML_FUNCTIONAL_HPP
#define TVTML_FUNCTIONAL_HPP

// Eigen includes
#include <Eigen/Core>
#include <Eigen/SparseCore>

// video++ includes
#include <vpp/vpp.hh>

// system includes
#include <cmath>
#include <iostream>
#include <fstream>

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
	static const int value_dim = MANIFOLD::value_type::SizeAtCompileTime;
	typedef typename MANIFOLD::scalar_type scalar_type;
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
	typedef double param_type;
	typedef double return_type;
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic,1> gradient_type;
	typedef Eigen::SparseMatrix<scalar_type> hessian_type;

	//Constructor
	Functional(param_type lambda, DATA dat):
	    lambda_(lambda),
	    data_(dat)
	{
	    eps2_=0.0;
	}
	
	// Evaluation functions
	return_type evaluateJ();
	void  evaluateDJ();
	
	void output_img(const img_type& img, const char* filename) const;

    private:
	param_type lambda_;
	param_type eps2_;
	DATA& data_;
	gradient_type DJ_;
	hessian_type  HJ_;
};


//--------Implementation FIRSTORDER, ISO-----/

// Evaluation of J
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
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    

    weights_mat X,Y;
    X = weights_mat(data_.img_.domain());
    Y = weights_mat(data_.img_.domain());

    // TODO: Try to replace omp loops
    // Horizontal Neighbours
    vpp::pixel_wise(X, N) | [&] (weights_type& x, auto& nbh) { x = MANIFOLD::dist_squared(nbh(0,0),nbh(0,1)); };
    #pragma omp parallel for
    for(int r=0; r< nr; r++) X(r,nc-1)=0.0;
    
    // Vertical Neighbours
    vpp::pixel_wise(Y, N) | [&] (weights_type& y, auto& nbh) { y = MANIFOLD::dist_squared(nbh(0,0),nbh(1,0)); };
    
    weights_type *lastrow = &Y(nr-1,0);
    for(int c=0; c< nc; c++) lastrow[c]=0.0;
	
	
        data_.output_weights(X,"XWeights.csv");
        data_.output_weights(Y,"YWeights.csv");
	

    // Compute IRLS Weights
    // TODO:	- Maybe put in different function
    //		- Calculation of weights and inverse weights could also be put in single pixel_wise
    
    
    auto g =  [&] (weights_type& iw, weights_type& x, weights_type& y) { return iw / std::sqrt(x+y+eps2_); };
    data_.weights_ = vpp::pixel_wise(data_.iweights_, X, Y) | g ;
    
   /* 
    for(int r=0; r<nr; r++){
	
	weights_type* w = &data_.weights_(r,0);
	weights_type* iw = &data_.iweights_(r, 0);
	weights_type* x = &X(r,0);
	weights_type* y = &Y(r,0);
	
	for(int c=0; c<nc; c++)
	    w[c] =  iw[c] / std::sqrt(x[c]+y[c]+eps2_);
    }
    

   */ 
    data_.output_weights(data_.weights_,"IRLS_Weights.csv");

    J2 = vpp::sum( vpp::pixel_wise(data_.weights_) | [&] (weights_type& w) {return 1.0/w;} );

	std::cout << "J1: " << J1 << std::endl;
	std::cout << "J2: " << J2 << std::endl;

    return 0.5 * J1 + lambda_* J2;
}

// Evaluation of J'
template < typename MANIFOLD, class DATA >
void Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::evaluateDJ(){

    output_img(data_.img_,"img.csv");
    vpp::fill(data_.weights_, 1.0); // Reset for Debugging

    img_type grad = img_type(data_.img_.domain());
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    
    //GRADIENT OF FIDELITY TERM
    if(data_.doInpaint()){
	auto f = [] (value_type& g, value_type& i, value_type& n, bool inp ) { MANIFOLD::deriv1x_dist_squared(i,n,g); g*=(1-inp); };
	vpp::pixel_wise(grad, data_.img_, data_.noise_img_, data_.inp_) | f;
    }
    else{
	auto f = [] (value_type& g,value_type& i, value_type& n) { MANIFOLD::deriv1x_dist_squared(i,n,g); };
	vpp::pixel_wise(grad, data_.img_, data_.noise_img_) | f;
    }
    
    //GRADIENT OF TV TERM

    // Neighbourhood box
    nbh_type N = nbh_type(data_.img_);

    // Subimage boxes
    vpp::box2d without_last_col(vpp::vint2(0,0), vpp::vint2(nr-1, nc-2)); // subdomain without last column
    vpp::box2d without_first_col(vpp::vint2(0,1), vpp::vint2(nr-1, nc-1)); // subdomain without first column
    vpp::box2d without_last_row(vpp::vint2(0,0), vpp::vint2(nr-2, nc-1)); // subdomain without last row
    vpp::box2d without_first_row(vpp::vint2(1,0), vpp::vint2(nr-1, nc-1)); // subdomain without first row

    // Horizontal derivatives and weighting
    // ... w.r.t. to first argument
    {
	img_type XD1 = img_type(data_.img_.domain());
	vpp::pixel_wise(XD1, data_.weights_, N) | [&] (value_type& x, weights_type& w, auto& nbh) { 
	    MANIFOLD::deriv1x_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	output_img(XD1,"XD1.csv");
	auto grad_subX1  = grad | without_last_col;
	auto deriv_subX1 = XD1 | without_last_col;
	vpp::pixel_wise(grad_subX1, deriv_subX1) | [&] (value_type& g, value_type& d) { g+=d*lambda_; };
    } // Memory for temporary XD1 gets deallocated after this scope

    // ... w.r.t. second argument
    {
	img_type XD2 = img_type(data_.img_.domain());
	vpp::pixel_wise(XD2, data_.weights_, N) | [&] (value_type& x, weights_type& w ,auto& nbh) { 
	   MANIFOLD::deriv1y_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	output_img(XD2,"XD2.csv");
	auto grad_subX2  = grad | without_first_col;
        auto deriv_subX2 = XD2 | without_last_col;
	vpp::pixel_wise(grad_subX2, deriv_subX2) | [&] (value_type& g, value_type& d) { g+=d*lambda_; };
    }


    // Vertical derivatives and weighting
    // ... w.r.t. first argument
    {
	img_type YD1 = img_type(data_.img_.domain());
	vpp::pixel_wise(YD1, data_.weights_, N) | [&] (value_type& y, weights_type& w, auto& nbh) { 
	    MANIFOLD::deriv1x_dist_squared(nbh(0,0), nbh(1,0), y); y*=w; };
	output_img(YD1,"YD1.csv");
	auto grad_subY1  = grad | without_last_row;
	auto deriv_subY1 = YD1 | without_last_row;
	vpp::pixel_wise(grad_subY1, deriv_subY1) | [&] (value_type& g, value_type& d) { g+=d*lambda_; };
    }

    // ... w.r.t second argument
    {
	img_type YD2 = img_type(data_.img_.domain());
	vpp::pixel_wise(YD2, data_.weights_, N) | [&] (value_type& y, weights_type& w, auto& nbh) { 
		MANIFOLD::deriv1y_dist_squared(nbh(0,0), nbh(1,0), y); y*=w; };
	output_img(YD2,"YD2.csv");
        auto grad_subY2  = grad | without_first_row;
	auto deriv_subY2 = YD2 | without_last_row;
        vpp::pixel_wise(grad_subY2, deriv_subY2) | [&] (value_type& g, value_type& d) { g+=d*lambda_; };
    }

    output_img(grad,"grad.csv");

    // Flatten to single gradient vector
    // TODO: Check if this can be also realized via Eigen::Map to the imageND data
    DJ_ = gradient_type::Zero(nr*nc*value_dim); 

    // flatten rowwise
    // TODO: Switch to rowise after Debug
    //vpp::pixel_wise(grad, grad.domain()) | [&] (value_type& p, vpp::vint2 coord) { DJ_.segment(3*(nc*coord[0]+coord[1]), value_dim) = p; };
    
    // flatten colwise (as in Matlab code)
    vpp::pixel_wise(grad, grad.domain()) | [&] (value_type& p, vpp::vint2 coord) { DJ_.segment(3*(coord[0]+nr*coord[1]), value_dim) = p; };

    std::fstream f;
    f.open("gradJ.csv",std::fstream::out);
    f << DJ_;
    f.close();

}


template < typename MANIFOLD, class DATA >
void Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::output_img(const img_type& img, const char* filename) const{
    int nr = img.nrows();
    int nc = img.ncols();

    std::fstream f;
    f.open(filename, std::fstream::out);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    for (int r=0; r<nr; r++){
	const value_type* cur = &img(r,0);
	for (int c=0; c<nc; c++){
	    f << cur[c].format(CommaInitFmt);
	    if(c != nc-1) f << ",";
	}
	f <<  std::endl;
    }
    f.close();
}


}// end namespace tvtml

#endif
