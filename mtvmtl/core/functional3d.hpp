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
	w = MANIFOLD::dist_squared(i, n);
    };

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...X neighbours " << std::endl;
    #endif
    
    // X Neighbours
    fill3d(weightsX_, 0.0);
    pixel_wise3d(calc_dist, weightsX_ | without_last_x, data_.img_ | without_last_x, data_.img_ | without_first_x );

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Reweighting" << std::endl;
	std::cout << "\t\t...Y neighbours" << std::endl;
    #endif

    // Y Neighbours
    fill3d(weightsY_, 0.0);
    pixel_wise3d(calc_dist, weightsY_ | without_last_y, data_.img_ | without_last_y, data_.img_ | without_first_y );

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Reweighting" << std::endl;
	std::cout << "\t\t...Z neighbours" << std::endl;
    #endif

    // Z Neighbours
    fill3d(weightsZ_, 0.0);
    pixel_wise3d(calc_dist, weightsZ_ | without_last_z, data_.img_ | without_last_z, data_.img_ | without_first_z );

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Reweighting" << std::endl;
    #endif

    if(disc==ISO){
	auto g =  [&] (const weights_type& ew, weights_type& x, weights_type& y, weights_type& z) { x = ew / std::sqrt(x + y + z + eps2_); y = x; z = x; };
	pixel_wise3d(g, data_.edge_weights_, weightsX_, weightsY_, weightsZ_);
    }
    else{
	auto g =  [&] (const weights_type& ew, weights_type& w) { w = ew / std::sqrt(w+eps2_); };
	pixel_wise3d(g, data_.edge_weights_, weightsX_);
	pixel_wise3d(g, data_.edge_weights_, weightsY_);
	pixel_wise3d(g, data_.edge_weights_, weightsZ_);
    }
}

// Update the Tangent space ONB
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::updateTMBase(){
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tUpdate tangent space basis..." << std::endl;
    #endif


   tm_base_mat_type T(data_.img_.domain());
   pixel_wise3d([&] (tm_base_type& t, const value_type& i) { MANIFOLD::tangent_plane_base(i,t); }, T, data_.img_);
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
	auto f = [&] (const value_type& i, const value_type& n, const bool inp ) { J1 += MANIFOLD::dist_squared(i,n)*(1-inp); };
	pixel_wise3d_nothreads(f, data_.img_, data_.noise_img_, data_.inp_);
    }
    else{
	auto f = [&] (const value_type& i, const value_type& n) { J1 += MANIFOLD::dist_squared(i,n); };
	pixel_wise3d_nothreads(f, data_.img_, data_.noise_img_);
    }

	updateWeights();

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...TV part." << std::endl;
    #endif

    if(disc==ISO)
	pixel_wise3d_nothreads([&] (const weights_type& w) { J2 += 1.0/w;} ,weightsX_);
    else
	pixel_wise3d_nothreads([&] (const weights_type& wx, const weights_type& wy, const weights_type& wz) { J2 += 1.0 / wx + 1.0 / wy+ 1.0 / wz;}, weightsX_, weightsY_, weightsZ_); 
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "J1: " << J1 << std::endl;
	std::cout << "J2: " << J2 << std::endl;
    #endif

    return 0.5 * J1 + lambda_* J2;
}

// Evaluation of J'
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::evaluateDJ(){
    
    img_type grad = img_type(data_.img_.domain());
    int ns = data_.img_.nslices();
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tGradient evaluation..." << std::endl;
	std::cout << "\t\t...Fidelity part" << std::endl;
    #endif
    //GRADIENT OF FIDELITY TERM
    if(data_.doInpaint()){
	auto f = [] (value_type& g, const value_type& i, const value_type& n, const bool inp ) { MANIFOLD::deriv1x_dist_squared(i,n,g); g*=(1-inp); };
	pixel_wise3d(f, grad, data_.img_, data_.noise_img_, data_.inp_);
    }
    else{
	auto f = [] (value_type& g, const value_type& i, const value_type& n) { MANIFOLD::deriv1x_dist_squared(i,n,g); };
	pixel_wise3d(f, grad, data_.img_, data_.noise_img_);
    }
    
    //GRADIENT OF TV TERM
    
    // Subimage boxes
    vpp::box3d without_last_x(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 1, nc - 2)); // subdomain without last xslice
    vpp::box3d without_last_y(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 2, nc - 1)); // subdomain without last yslice
    vpp::box3d without_last_z(vpp::vint3(0,0,0), vpp::vint3(ns - 2, nr - 1, nc - 1)); // subdomain without last zslice
    vpp::box3d without_first_x(vpp::vint3(0,0,1), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first xlice
    vpp::box3d without_first_y(vpp::vint3(0,1,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first yslice
    vpp::box3d without_first_z(vpp::vint3(1,0,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first zslice

    auto calc_first_arg_deriv = [&] (value_type& x, const weights_type& w, const value_type& i, const value_type& n) { MANIFOLD::deriv1x_dist_squared(i, n, x); x *= w; };
    auto calc_second_arg_deriv = [&] (value_type& y, const weights_type& w, const value_type& i, const value_type& n) { MANIFOLD::deriv1y_dist_squared(i, n, y); y *= w; };
    auto add_to_gradient = [&] (value_type& g, const value_type& d) { g+=d*lambda_; };

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tGradient evaluation..." << std::endl;
	std::cout << "\t\t...TV part" << std::endl;
	std::cout << "\t\t...-> XD1" << std::endl;
    #endif
    // X neighbors and reweighting
    // ... w.r.t. to first argument
    { // Temporary image XD1 is deallocated after this scope 
	img_type XD1 = img_type(without_last_x);
	pixel_wise3d(calc_first_arg_deriv, XD1, weightsX_ | without_last_x, data_.img_ | without_last_x, data_.img_ | without_first_x);
	pixel_wise3d(add_to_gradient, grad | without_last_x, XD1);
    }
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...-> XD2" << std::endl;
    #endif
    // ... w.r.t. second argument
    {
	img_type XD2 = img_type(without_last_x);
	pixel_wise3d(calc_second_arg_deriv, XD2, weightsX_ | without_last_x, data_.img_ | without_last_x, data_.img_ | without_first_x);
	pixel_wise3d(add_to_gradient, grad | without_first_x,  XD2);
    }

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...-> YD1" << std::endl;
    #endif
    // Vertical derivatives and weighting
    // ... w.r.t. first argument
    {
	img_type YD1 = img_type(without_last_y);
	pixel_wise3d(calc_first_arg_deriv, YD1, weightsY_ | without_last_y, data_.img_ | without_last_y, data_.img_ | without_first_y);
	pixel_wise3d(add_to_gradient, grad | without_first_y, YD1);
    }
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...-> YD2" << std::endl;
    #endif
    // ... w.r.t second argument
    {
	img_type YD2 = img_type(without_last_y);
	pixel_wise3d(calc_second_arg_deriv, YD2, weightsY_ | without_last_y, data_.img_ | without_last_y, data_.img_ | without_first_y);
        pixel_wise3d(add_to_gradient, grad | without_first_y, YD2);
    }

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...-> ZD1" << std::endl;
    #endif
    // Vertical derivatives and weighting
    // ... w.r.t. first argument
    {
	img_type ZD1 = img_type(without_last_z);
	pixel_wise3d(calc_first_arg_deriv, ZD1, weightsZ_ | without_last_z, data_.img_ | without_last_z, data_.img_ | without_first_z);
	pixel_wise3d(add_to_gradient, grad | without_first_y, ZD1);
    }
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...-> ZD2" << std::endl;
    #endif
    // ... w.r.t second argument
    {
	img_type ZD2 = img_type(without_last_z);
	pixel_wise3d(calc_second_arg_deriv, ZD2, weightsZ_ | without_last_z, data_.img_ | without_last_z, data_.img_ | without_first_z);
        pixel_wise3d(add_to_gradient, grad | without_first_z, ZD2);
    }
    
    DJ_ = gradient_type::Zero(ns*nr*nc*manifold_dim); 
    
    updateTMBase();
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Local to global insert" << std::endl;
    #endif
    
   for(int s = 0; s < ns; ++s){
	//#pragma omp parallel for
	for(int r = 0; r < nr; ++r){
	    // Start of row pointers
	    value_type* p = &grad(s, r, 0);
	    tm_base_type* t = &T_(s, r, 0);
	    for(int c = 0; c < nc; ++c)
		DJ_.segment(manifold_dim * (ns * nc * s + nc * r + c), manifold_dim) = t[c].transpose() * Eigen::Map<const Eigen::VectorXd>(p[c].data(), p[c].size()); 
	//	DJ_.segment(manifold_dim * (s + ns * r + ns * nr * c), manifold_dim) = t[c].transpose() * Eigen::Map<const Eigen::VectorXd>(p[c].data(), p[c].size()); 
	}
    } 

    #ifdef TV_FUNC_DEBUG 
	std::fstream f;
	f.open("gradJ.csv",std::fstream::out);
	f << DJ_;
	f.close();
    #endif
}

// Evaluation of Hessian J
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA, 3 >::evaluateHJ(){
    hessian_type hessian(data_.img_.domain());

    int ns = data_.img_.nslices();
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    int sparsedim = ns*nr*nc*manifold_dim;
    
        
    //HESSIAN OF FIDELITY TERM
     #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tHessian evaluation..." << std::endl;
	std::cout << "\t\t...Fidelity part" << std::endl;
    #endif

    sparse_hessian_type HF(sparsedim,sparsedim);

    //HF.reserve(Eigen::VectorXi::Constant(nc,manifold_dim));
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> triplist;
    triplist.reserve(sparsedim*manifold_dim);
	
    if(data_.doInpaint()){
	auto f = [] (deriv2_type& h, const value_type& i, const value_type& n, const bool inp ) { MANIFOLD::deriv2xx_dist_squared(i,n,h); h*=(1-inp); };
	pixel_wise3d(f, hessian, data_.img_, data_.noise_img_, data_.inp_);
    }
    else{
	auto f = [] (deriv2_type& h, const value_type& i, const value_type& n) { MANIFOLD::deriv2xx_dist_squared(i,n,h); };
	pixel_wise3d(f, hessian, data_.img_, data_.noise_img_);
    }
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Local to global insert" << std::endl;
    #endif
   for(int s = 0; s < ns; ++s){
    for(int r = 0; r < nr; ++r){
	    // Start of row pointers
	    deriv2_type* h = &hessian(s, r, 0);
	    tm_base_type* t = &T_(s, r, 0);
	    for(int c = 0; c < nc; ++c){
		int pos = manifold_dim * (ns * nc * s + nc * r + c); // rowwise flattening
		//int pos = manifold_dim * (s + ns * r + ns * nr * c); // columnwise flattening
		restricted_deriv2_type ht=t[c].transpose()*h[c]*t[c];
	    
		for(int local_row = 0; local_row<ht.rows(); local_row++)
		    for(int local_col = local_row; local_col < ht.cols(); local_col++){
			scalar_type e = ht(local_row, local_col);
			    if(e!=0){
				int global_row = pos + local_row;
				int global_col = pos + local_col;
				triplist.push_back(Trip(global_row,global_col,e));
				if(global_row != global_col)
				    triplist.push_back(Trip(global_col,global_row,e));
			    }
		    }
	    
	    }
	}
    } 
	
     #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Triplet list created" << std::endl;
    #endif
    HF.setFromTriplets(triplist.begin(),triplist.end());              
    HF.makeCompressed();

    //HESSIAN OF TV TERM
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tHessian evaluation..." << std::endl;
	std::cout << "\t\t...TV part" << std::endl;
    #endif

    sparse_hessian_type HTV(sparsedim,sparsedim);
    
    //HTV.reserve(Eigen::VectorXi::Constant(nc,5*manifold_dim));
    triplist.clear();
    triplist.reserve(5 * sparsedim*manifold_dim);

    // Subimage boxes
    vpp::box3d without_last_x(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 1, nc - 2)); // subdomain without last xslice
    vpp::box3d without_last_y(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 2, nc - 1)); // subdomain without last yslice
    vpp::box3d without_last_z(vpp::vint3(0,0,0), vpp::vint3(ns - 2, nr - 1, nc - 1)); // subdomain without last zslice
    vpp::box3d without_first_x(vpp::vint3(0,0,1), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first xlice
    vpp::box3d without_first_y(vpp::vint3(0,1,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first yslice
    vpp::box3d without_first_z(vpp::vint3(1,0,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first zslice


    auto calc_xx_der = [&] (deriv2_type& x, const weights_type& w, const value_type& i, const value_type& n) { MANIFOLD::deriv2xx_dist_squared(i, n, x); x*=w; };
    auto calc_xy_der = [&] (deriv2_type& xy, const weights_type& w, const value_type& i, const value_type& n) { MANIFOLD::deriv2xy_dist_squared(i, n, xy); xy*=w; };
    auto calc_yy_der = [&] (deriv2_type& y, const weights_type& w, const value_type& i, const value_type& n) { MANIFOLD::deriv2yy_dist_squared(i, n, y); y*=w; };
    auto add_to_hessian =  [&] (deriv2_type& h, const deriv2_type& d) { h += d; };

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->XD11" << std::endl;
    #endif
    
    // Horizontal Second Derivatives and weighting
    // ... w.r.t. first arguments
    { // Temporary image XD11 is deallocated after this scope
	hessian_type XD11(without_last_x);
        pixel_wise3d(calc_xx_der, XD11, weightsX_ | without_last_x, data_.img_ | without_last_x, data_.img_ | without_first_x);
	pixel_wise3d([&] (deriv2_type& h, const deriv2_type& d) { h=d; }, hessian | without_last_x, XD11);
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(XD11,"3dXD11.csv");
	#endif
    }
    
    for(int s=0; s < ns; ++s)
	for(int r=0; r<nr; ++r)
	    hessian(s,r,nc-1)=deriv2_type::Zero();


    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->XD22" << std::endl;
    #endif
    //... w.r.t. second arguments
    {
	hessian_type XD22(without_last_x);
        pixel_wise3d(calc_yy_der, XD22, weightsX_ | without_last_x, data_.img_ | without_last_x, data_.img_ | without_first_x);
	pixel_wise3d(add_to_hessian, hessian | without_first_x, XD22);
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(XD22,"3dXD22.csv");
	#endif
    }
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->YD11" << std::endl;
    #endif
    // Vertical Second Derivatives weighting
    //... w.r.t. first arguments
    {
	hessian_type YD11(without_last_y);
        pixel_wise3d(calc_xx_der, YD11, weightsY_ | without_last_y, data_.img_ | without_last_y, data_.img_ | without_first_y);
	pixel_wise3d(add_to_hessian, hessian | without_last_y, YD11);
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(YD11,"3dYD11.csv");
	#endif
    }

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->YD22" << std::endl;
    #endif
    //... w.r.t. second arguments
    {
	hessian_type YD22(without_last_y);
        pixel_wise3d(calc_yy_der, YD22, weightsY_ | without_last_y, data_.img_ | without_last_y, data_.img_ | without_first_y); 
	pixel_wise3d(add_to_hessian, hessian | without_first_y, YD22);
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(YD22,"3dYD22.csv");
	#endif
    }
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->ZD11" << std::endl;
    #endif
    // Vertical Second Derivatives weighting
    //... w.r.t. first arguments
    {
	hessian_type ZD11(without_last_z);
        pixel_wise3d(calc_xx_der, ZD11, weightsZ_ | without_last_z, data_.img_ | without_last_z, data_.img_ | without_first_z);
	pixel_wise3d(add_to_hessian, hessian | without_last_z, ZD11);
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(ZD11,"3dZD11.csv");
	#endif
    }

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->ZD22" << std::endl;
    #endif
    //... w.r.t. second arguments
    {
	hessian_type ZD22(without_last_z);
        pixel_wise3d(calc_yy_der, ZD22, weightsZ_ | without_last_z, data_.img_ | without_last_z, data_.img_ | without_first_z); 
	pixel_wise3d(add_to_hessian, hessian | without_first_z, ZD22);
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(ZD22,"3dZD22.csv");
	#endif
    } 
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Local to global insert" << std::endl;
    #endif
    // Insert elementwise into sparse Hessian
    // NOTE: Eventually make single version for both cases, including an offset
    // --> additional parameters sparse_mat, offset
    auto HTV_insert = [&]( hessian_type H, tm_base_mat_type T1, tm_base_mat_type T2, int row_offset, int col_offset){
	int Hns = H.nslices();
	int Hnr = H.nrows();
	int Hnc = H.ncols();
    
	for(int s = 0; s < Hns; ++s){
	    for(int r = 0; r < Hnr; ++r){
		// Start of row pointers
		deriv2_type* h = &H(s, r, 0);
		tm_base_type* t1 = &T1(s, r, 0);
		tm_base_type* t2 = &T2(s, r, 0);
		for(int c = 0; c < Hnc; ++c){
		    int pos = manifold_dim * (ns * nc * s + nc * r +c); // rowwise flattening
		   // int pos = manifold_dim * (s + ns * r + ns * nr * c); // columnwise flattening
		    restricted_deriv2_type ht = t1[c].transpose()*h[c]*t2[c];
		
		    for(int local_row = 0; local_row<ht.rows(); local_row++)
			for(int local_col = 0; local_col < ht.cols(); local_col++){
			    scalar_type e = ht(local_row, local_col);
				if(e!=0){
				    int global_row = pos + row_offset + local_row;
				    int global_col = pos + col_offset + local_col;
				    triplist.push_back(Trip(global_row,global_col,e));
				    if(row_offset > 0 || col_offset > 0)
					triplist.push_back(Trip(global_col,global_row,e));
				}
			}
		
		}
	    }
	}
    };

    auto HTV_single_insert = [&] (deriv2_type& h, tm_base_type& t1, tm_base_type& t2, int pos, int row_offset, int col_offset){
   		    restricted_deriv2_type ht = t1.transpose()*h*t2;
		    for(int local_row = 0; local_row<ht.rows(); local_row++)
			for(int local_col = 0; local_col < ht.cols(); local_col++){
			    scalar_type e = ht(local_row, local_col);
				if(e!=0){
				    int global_row = pos + row_offset + local_row;
				    int global_col = pos + col_offset + local_col;
				    triplist.push_back(Trip(global_row,global_col,e));
				    if(row_offset > 0 || col_offset > 0)
					triplist.push_back(Trip(global_col,global_row,e));
				}
			}
    };

    HTV_insert(hessian, T_, T_, 0, 0);

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->ZD12" << std::endl;
    #endif              
    // Z Neighbors Second Derivatives and weighting
    // ... w.r.t. first and second arguments 
    {
	hessian_type ZD12(without_last_z);
	pixel_wise3d(calc_xy_der, ZD12, weightsZ_ | without_last_z, data_.img_ | without_last_z, data_.img_ | without_first_z );
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(ZD12,"3dZD12.csv");
	#endif
	#ifdef TV_FUNC_DEBUG_VERBOSE
		std::cout << "\t\t...Local to global insert:" << std::endl;
	#endif
	// Offsets for upper nx*ny-th subdiagonal
	int offset =  manifold_dim * nr * nc;
	HTV_insert(ZD12, T_ | without_last_z, T_ | without_first_z, 0, offset);
    }
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->YD12" << std::endl;
    #endif
    // Y Neighbors Second Derivatives and weighting
    //... w.r.t. second arguments
    {
	hessian_type YD12(data_.img_.domain());
	pixel_wise3d(calc_xy_der, YD12 | without_last_y, weightsY_ | without_last_y, data_.img_ | without_last_y, data_.img_ | without_first_y );
	
	//Set last y-slice to zero
	#pragma omp parallel for
	for(int s = 0; s < ns ; ++s){
	    deriv2_type* row = &YD12(s, nr-1 ,0);
	    for(int c = 0; c < nc; ++c)
		row[c] = deriv2_type::Zero();
	}
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(YD12,"3dYD12.csv");
	#endif
	#ifdef TV_FUNC_DEBUG_VERBOSE
		std::cout << "\t\t...Local to global insert:" << std::endl;
	#endif
	// Offsets for first nxth subdiagonal
	int offset =  manifold_dim * nc;

	// Insert YD12 except last ns entries
	auto t1_it = T_.begin();
	auto t2_it = typename tm_base_mat_type::iterator(vpp::vint3(0,1,0), T_); // Iterator at beginning of second row 
	bool do_break = false;
	int k = 0;
	int max_entry = ns * nr * nc - nc;

	for(int s=0; s < ns; ++s){
	    for(int r=0; r < nr; ++r){
		 deriv2_type* row = &YD12(s, r, 0);
		 for(int c=0; c < nc; ++c){
		    int pos = manifold_dim * (ns * nc * s + nc * r +c); // rowwise flattening
	//	    int pos = manifold_dim * (s + ns * r + ns * nr * c); // columnwise flattening
		    HTV_single_insert(row[c], *t1_it, *t2_it, pos, 0, offset);
		    t1_it.next();
		    t2_it.next();
		    ++k;
		    if(k == max_entry) {do_break = true; break;}
		 }
		 if(do_break) break;
	    }
	    if(do_break) break;
	}


    }

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->XD12" << std::endl;
    #endif
    // Z neightbors Second Derivatives and weighting
    //... w.r.t. second arguments
    {
	hessian_type XD12(data_.img_.domain());
	pixel_wise3d(calc_xy_der, XD12 | without_last_x, weightsX_ | without_last_x, data_.img_ | without_last_x, data_.img_ | without_first_x);
	
	//Set last slice to zero, - actually not necessary but safer than keeping uninitialized data
	#pragma omp parallel for
	for(int s = 0; s < ns; ++s){
	    for(int r = 0; r < nr; ++r)
		XD12(s, r, nc-1) = deriv2_type::Zero();
	}
	
	#ifdef TV_FUNC_DEBUG
	    data_.output_matval_img(XD12,"3dXD12.csv");
	#endif
	#ifdef TV_FUNC_DEBUG_VERBOSE
		std::cout << "\t\t...Local to global insert:" << std::endl;
	#endif
	
	// Insert into first upper subdiagonal
	auto t1_it = T_.begin();
	auto t2_it = T_.begin(); t2_it.next();
	for(int s=0; s < ns; ++s){ 
	    for(int r=0; r < nr; ++r){
		 deriv2_type* row = &XD12(s, r, 0);
		 for(int c=0; c < nc; ++c){
		    int pos = manifold_dim * (ns * nc * s + nc * r +c); // rowwise flattening
		    //int pos = manifold_dim * (s + ns * r + ns * nr * c); // columnwise flattening
		    HTV_single_insert(row[c], *t1_it, *t2_it, pos, 0, manifold_dim);
		    t1_it.next();
		    t2_it.next();
		    if(s==ns-1 && r == nc-1 && c >= nc-2 ) break;
		 }
	    }
	}

	HTV.setFromTriplets(triplist.begin(),triplist.end());              
	HTV.makeCompressed();
	triplist.clear();
    }   	
	#ifdef TV_FUNC_DEBUG_VERBOSE
		std::cout << "\t\t...Combine Fidelity and TV parts:" << std::endl;
	#endif 
	
	HJ_= HF + lambda_*HTV;
	
	#ifdef TV_FUNC_DEBUG_VERBOSE
		std::cout << "\t\t...Output Hessian (stats):" << std::endl;
	#endif
    #ifdef TV_FUNC_DEBUG
	if (sparsedim<200){
	    if(sparsedim<80){
		std::cout << "\nFidelity\n" << HF << std::endl; 
		std::cout << "\nTV\n" << HTV << std::endl; 
		std::cout << "\nHessian\n" << HJ_ << std::endl; 
	    }

	    std::fstream f;
	    f.open("H.csv",std::fstream::out);
	    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
	    //f << HJ_.format(CommaInitFmt).;
	    f << HJ_;
	    f.close();

	}
	else{
	    std::cout << "\nFidelity Non-Zeros: " << HJ_.nonZeros() << std::endl; 
	    std::cout << "\nTV Non-Zeros: " << HJ_.nonZeros() << std::endl; 
	    std::cout << "\nHessian Non-Zeros: " << HJ_.nonZeros() << std::endl; 
	}
    // Test Solver:
	#ifdef TV_FUNC_DEBUG_VERBOSE
		std::cout << "\t\t...Test Solve" << std::endl;
	#endif
	gradient_type x;
    
	Eigen::SparseLU<sparse_hessian_type> solver;
	solver.analyzePattern(HJ_);
	solver.factorize(HJ_);
	x = solver.solve(DJ_);

	std::fstream f;
	f.open("Sol.csv",std::fstream::out);
	f << x;
	f.close();
    #endif
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
