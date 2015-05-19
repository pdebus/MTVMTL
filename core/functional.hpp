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
	typedef typename MANIFOLD::deriv1_type deriv1_type;
	typedef typename MANIFOLD::deriv2_type deriv2_type;

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
	typedef vpp::imageNd<deriv2_type, img_dim> hessian_type;
	typedef Eigen::SparseMatrix<scalar_type> sparse_hessian_type;

	//Constructor
	Functional(param_type lambda, DATA dat):
	    lambda_(lambda),
	    data_(dat)
	{
	    eps2_=0.0;
    	}
	
	// Evaluation functions
	void updateWeights();
	return_type evaluateJ();
	void  evaluateDJ();
	void  evaluateHJ();
	
	void output_img(const img_type& img, const char* filename) const;

    private:
	param_type lambda_;
	param_type eps2_;
	DATA& data_;
	gradient_type DJ_;
	sparse_hessian_type HJ_;
};


//--------Implementation FIRSTORDER, ISO-----/


// Update the Weights
template < typename MANIFOLD, class DATA >
void Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::updateWeights(){

    // Neighbourhood box
    nbh_type N = nbh_type(data_.img_); 
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();

    weights_mat X,Y;
    X = weights_mat(data_.img_.domain());
    Y = weights_mat(data_.img_.domain());

    // TODO: Try to replace omp loops e.g. by box2d (nr-1,0) (nr-1,nc-1) or Iterator
    // Horizontal Neighbours
    vpp::pixel_wise(X, N) | [&] (weights_type& x, const auto& nbh) { x = MANIFOLD::dist_squared(nbh(0,0),nbh(0,1)); };
    #pragma omp parallel for
    for(int r=0; r< nr; r++) 
	X(r,nc-1)=0.0;
    
    // Vertical Neighbours
    vpp::pixel_wise(Y, N) | [&] (weights_type& y, const auto& nbh) { y = MANIFOLD::dist_squared(nbh(0,0),nbh(1,0)); };
    weights_type *lastrow = &Y(nr-1,0);
    for(int c=0; c< nc; c++) 
	lastrow[c]=0.0;
	
    //data_.output_weights(X,"XWeights.csv");
    //data_.output_weights(Y,"YWeights.csv");
	
    
    auto g =  [&] (weights_type& w, const weights_type& iw, const weights_type& x, const weights_type& y) { w = iw / std::sqrt(x+y+eps2_); };
    vpp::pixel_wise(data_.weights_, data_.iweights_, X, Y) | g ;
    
    //data_.output_weights(data_.weights_,"IRLS_Weights.csv");
}


// Evaluation of J
template < typename MANIFOLD, class DATA >
typename Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::return_type Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::evaluateJ(){

    // sum d^2(img, img_noise)
    return_type J1, J2;
    J1 = J2 = 0.0;

    if(data_.doInpaint()){
	auto f = [] (const value_type& i, const value_type& n, const bool inp ) { return MANIFOLD::dist_squared(i,n)*(1-inp); };
	J1 = vpp::sum(vpp::pixel_wise(data_.img_, data_.noise_img_, data_.inp_) | f);
    }
    else{
	auto f = [] (const value_type& i, const value_type& n) { return MANIFOLD::dist_squared(i,n); };
	J1 = vpp::sum(vpp::pixel_wise(data_.img_, data_.noise_img_) | f);
    }

    updateWeights();

    J2 = vpp::sum( vpp::pixel_wise(data_.weights_) | [&] (const weights_type& w) {return 1.0/w;} );

	//std::cout << "J1: " << J1 << std::endl;
	//std::cout << "J2: " << J2 << std::endl;

    return 0.5 * J1 + lambda_* J2;
}

// Evaluation of J'
template < typename MANIFOLD, class DATA >
void Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::evaluateDJ(){

    //output_img(data_.img_,"img.csv");
    vpp::fill(data_.weights_, 1.0); // Reset for Debugging

    img_type grad = img_type(data_.img_.domain());
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    
    //GRADIENT OF FIDELITY TERM
    if(data_.doInpaint()){
	auto f = [] (value_type& g, const value_type& i, const value_type& n, const bool inp ) { MANIFOLD::deriv1x_dist_squared(i,n,g); g*=(1-inp); };
	vpp::pixel_wise(grad, data_.img_, data_.noise_img_, data_.inp_) | f;
    }
    else{
	auto f = [] (value_type& g, const value_type& i, const value_type& n) { MANIFOLD::deriv1x_dist_squared(i,n,g); };
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
    { // Temporary image XD1 is deallocated after this scope 
	img_type XD1 = img_type(data_.img_.domain());
	vpp::pixel_wise(XD1, data_.weights_, N) | [&] (value_type& x, const weights_type& w, const auto& nbh) { 
	    MANIFOLD::deriv1x_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	output_img(XD1,"XD1.csv");
	auto grad_subX1  = grad | without_last_col;
	auto deriv_subX1 = XD1 | without_last_col;
	vpp::pixel_wise(grad_subX1, deriv_subX1) | [&] (value_type& g, const value_type& d) { g+=d*lambda_; };
    } 
    // ... w.r.t. second argument
    {
	img_type XD2 = img_type(data_.img_.domain());
	vpp::pixel_wise(XD2, data_.weights_, N) | [&] (value_type& x, const weights_type& w ,const auto& nbh) { 
	   MANIFOLD::deriv1y_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	output_img(XD2,"XD2.csv");
	auto grad_subX2  = grad | without_first_col;
        auto deriv_subX2 = XD2 | without_last_col;
	vpp::pixel_wise(grad_subX2, deriv_subX2) | [&] (value_type& g, const value_type& d) { g+=d*lambda_; };
    }


    // Vertical derivatives and weighting
    // ... w.r.t. first argument
    {
	img_type YD1 = img_type(data_.img_.domain());
	vpp::pixel_wise(YD1, data_.weights_, N) | [&] (value_type& y, const weights_type& w, const auto& nbh) { 
	    MANIFOLD::deriv1x_dist_squared(nbh(0,0), nbh(1,0), y); y*=w; };
	output_img(YD1,"YD1.csv");
	auto grad_subY1  = grad | without_last_row;
	auto deriv_subY1 = YD1 | without_last_row;
	vpp::pixel_wise(grad_subY1, deriv_subY1) | [&] (value_type& g, const value_type& d) { g+=d*lambda_; };
    }

    // ... w.r.t second argument
    {
	img_type YD2 = img_type(data_.img_.domain());
	vpp::pixel_wise(YD2, data_.weights_, N) | [&] (value_type& y, const weights_type& w, const auto& nbh) { 
		MANIFOLD::deriv1y_dist_squared(nbh(0,0), nbh(1,0), y); y*=w; };
	output_img(YD2,"YD2.csv");
        auto grad_subY2  = grad | without_first_row;
	auto deriv_subY2 = YD2 | without_last_row;
        vpp::pixel_wise(grad_subY2, deriv_subY2) | [&] (value_type& g, const value_type& d) { g+=d*lambda_; };
    }

    output_img(grad,"grad.csv");

    // Flatten to single gradient vector
	// TODO: Check if this can be also realized via Eigen::Map to the imageND data
    DJ_ = gradient_type::Zero(nr*nc*value_dim); 

    // flatten rowwise
	// TODO: Switch to rowise after Debug
    //vpp::pixel_wise(grad, grad.domain()) | [&] (value_type& p, vpp::vint2 coord) { DJ_.segment(3*(nc*coord[0]+coord[1]), value_dim) = p; };
    
    // flatten colwise (as in Matlab code)
    vpp::pixel_wise(grad, grad.domain()) | [&] (const value_type& p, const vpp::vint2 coord) { DJ_.segment(3*(coord[0]+nr*coord[1]), value_dim) = p; };

    
    std::fstream f;
    f.open("gradJ.csv",std::fstream::out);
    f << DJ_;
    f.close();
    
}

// Evaluation of Hessian J
template < typename MANIFOLD, class DATA >
void Functional< FIRSTORDER, ISO, MANIFOLD, DATA >::evaluateHJ(){
    vpp::fill(data_.weights_, 1.0); // Reset for Debugging

    hessian_type hessian(data_.img_.domain());
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    int sparsedim = nr*nc*value_dim;
    
   
        
    //HESSIAN OF FIDELITY TERM

    sparse_hessian_type HF(sparsedim,sparsedim);
    HF.reserve(Eigen::VectorXi::Constant(nc,value_dim));
	
    if(data_.doInpaint()){
	auto f = [] (deriv2_type& h, const value_type& i, const value_type& n, const bool inp ) { MANIFOLD::deriv2xx_dist_squared(i,n,h); h*=(1-inp); };
	vpp::pixel_wise(hessian, data_.img_, data_.noise_img_, data_.inp_) | f;
    }
    else{
	auto f = [] (deriv2_type& h, const value_type& i, const value_type& n) { MANIFOLD::deriv2xx_dist_squared(i,n,h); };
	vpp::pixel_wise(hessian, data_.img_, data_.noise_img_) | f;
    }
    
   //TODO: Check whether all 2nd-derivative matrices are symmetric s.t. only half the matrix need to be traversed. e.g. local_col=local_row instead of 0
    auto local2globalInsert = [&](deriv2_type& h, vpp::vint2 coord) { 
	int pos = 3*(coord[0]+nr*coord[1]); // columnwise flattening
	for(int local_row=0; local_row<h.rows(); local_row++)
	    for(int local_col=0; local_col<h.cols(); local_col++)
		if(h(local_row, local_col)!=0)
		    HF.insert(pos+local_row, pos+local_col) = h(local_row, local_col);
    };

    vpp::pixel_wise(hessian, hessian.domain())(/*vpp::_no_threads*/) | local2globalInsert;
                   
    if (sparsedim<70)
        std::cout << "HF:\n" << HF << std::endl; 
    else
        std::cout << "HF Non-Zeros: " << HF.nonZeros() << std::endl; 

    
    //HESSIAN OF TV TERM
    sparse_hessian_type HTV(sparsedim,sparsedim);
    HTV.reserve(Eigen::VectorXi::Constant(nc,value_dim));
	
    // Neighbourhood box
    nbh_type N = nbh_type(data_.img_);
    
    // Subimage boxes
    // TODO: Make static class variables
    vpp::box2d without_last_col(vpp::vint2(0,0), vpp::vint2(nr-1, nc-2)); // subdomain without last column
    vpp::box2d without_first_col(vpp::vint2(0,1), vpp::vint2(nr-1, nc-1)); // subdomain without first column
    vpp::box2d without_last_row(vpp::vint2(0,0), vpp::vint2(nr-2, nc-1)); // subdomain without last row
    vpp::box2d without_first_row(vpp::vint2(1,0), vpp::vint2(nr-1, nc-1)); // subdomain without first row

    // Horizontal Second Derivatives and weighting
    // ... w.r.t. first arguments
    { // Temporary image XD11 is deallocated after this scope
	hessian_type XD11(data_.img_.domain());
        vpp::pixel_wise(XD11, data_.weights_, N) | [&] (deriv2_type& x, const weights_type& w, const auto& nbh) { 
    	    MANIFOLD::deriv2xx_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	//output_img(XD11,"XD11.csv");
	auto hess_subX11  = hessian | without_last_col;
	auto deriv_subX11 = XD11 | without_last_col;
	vpp::pixel_wise(hess_subX11, deriv_subX11) | [&] (deriv2_type& h, const deriv2_type& d) { h=d; };
    }
	#pragma omp parallel for
	for(int r=0; r< nr; r++) 
	    hessian(r,nc-1)=deriv2_type::Zero(); // set last column to zero
    
    //... w.r.t. second arguments
    {
	hessian_type XD22(data_.img_.domain());
	vpp::pixel_wise(XD22, data_.weights_, N) | [&] (deriv2_type& x, const weights_type& w, const auto& nbh) { 
	    MANIFOLD::deriv2yy_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	//output_img(XD22,"XD22.csv");
	auto hess_subX22  = hessian | without_first_col;
	auto deriv_subX22 = XD22 | without_first_col;
	vpp::pixel_wise(hess_subX22, deriv_subX22) | [&] (deriv2_type& h, const deriv2_type& d) { h+=d; };
    }
    // Vertical Second Derivatives weighting
    //... w.r.t. first arguments
    {
	hessian_type YD11(data_.img_.domain());
	vpp::pixel_wise(YD11, data_.weights_, N) | [&] (deriv2_type& x, const weights_type& w, const auto& nbh) { 
	    MANIFOLD::deriv2xx_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	//output_img(YD11,"YD11.csv");
	auto hess_subY11  = hessian | without_last_row;
	auto deriv_subY11 = YD11 | without_last_row;
	vpp::pixel_wise(hess_subY11, deriv_subY11) | [&] (deriv2_type& h, const deriv2_type& d) { h+=d; };
    }
    //... w.r.t. second arguments
    {
	hessian_type YD22(data_.img_.domain());
	vpp::pixel_wise(YD22, data_.weights_, N) | [&] (deriv2_type& x, const weights_type& w, const auto& nbh) { 
	        MANIFOLD::deriv2xx_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
        //output_img(YD22,"YD22.csv");
	auto hess_subY22  = hessian | without_first_row;
	auto deriv_subY22 = YD22 | without_first_row;
	vpp::pixel_wise(hess_subY22, deriv_subY22) | [&] (deriv2_type& h, const deriv2_type& d) { h+=d; };
    }
    // Insert elementwise into sparse Hessian
    // NOTE: Eventually make single version for both cases, including an offset
    // --> additional parameters sparse_mat, offset
    int row_offset=0;
    int col_offset=0;
    auto local2globalInsertHTV = [&](const deriv2_type& h, const vpp::vint2 coord) { 
	int pos = 3*(coord[0]+nr*coord[1]); // columnwise flattening
	for(int local_row=0; local_row<h.rows(); local_row++)
	    for(int local_col=0; local_col<h.cols(); local_col++)
		if(h(local_row, local_col)!=0)
		    HTV.insert(pos + row_offset + local_row, pos + col_offset + local_col) = h(local_row, local_col);
    };
    vpp::pixel_wise(hessian, hessian.domain())(/*vpp::_no_threads*/) | local2globalInsertHTV;
                   
    // Horizontal Second Derivatives and weighting
    // ... w.r.t. first and second arguments 
    {
	hessian_type XD12(without_last_col);
	vpp::pixel_wise(XD12, data_.weights_ | without_last_col, N) | [&] (deriv2_type& x, const weights_type& w, const auto& nbh) { 
	    MANIFOLD::deriv2xy_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	//output_img(XD12,"XD12.csv");

	// Offsets for upper nyth subdiagonal
	row_offset=0;
	col_offset=value_dim*nr;
	vpp::pixel_wise(XD12, XD12.domain())(/*vpp::_no_threads*/) | local2globalInsertHTV;
    }
    // Vertical Second Derivatives and weighting
    //... w.r.t. second arguments
    {
	hessian_type YD12(data_.img_.domain());
	vpp::pixel_wise(YD12, data_.weights_, N) | [&] (deriv2_type& x, const weights_type& w, const auto& nbh) { 
	    MANIFOLD::deriv2xy_dist_squared(nbh(0,0), nbh(0,1), x); x*=w; };
	//output_img(YD12,"YD12.csv");
	//Set last row to zero
	deriv2_type *lastrow = &YD12(nr-1,0);
	for(int c=0; c< nc; c++) 
	    lastrow[c]=deriv2_type::Zero();
    
	// Offsets for first upper subdiagonal
	row_offset=0;
	col_offset=value_dim;
	vpp::pixel_wise(YD12 | without_last_row, without_last_row)(/*vpp_no_threads*/) | local2globalInsertHTV;
	for(int c=0; c<nc-1; c++) 
	    local2globalInsertHTV(lastrow[c], vpp::vint2(nr-1,c));
    }

    if (sparsedim<70)
        std::cout << "HTV\n" << HTV << std::endl; 
    else
        std::cout << "HTV Non-Zeros: " << HTV.nonZeros() << std::endl; 

    HJ_= HF + lambda_*HTV;

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
