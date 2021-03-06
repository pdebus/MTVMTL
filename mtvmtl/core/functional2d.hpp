#ifndef TVTML_FUNCTIONAL2D_HPP
#define TVTML_FUNCTIONAL2D_HPP

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

namespace tvmtl{


template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA>
class Functional<FIRSTORDER, disc, MANIFOLD, DATA>{

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
	   static_assert(img_dim == 2, "Dimension of data and functional must match!");
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

	inline const gradient_type& getDJ() const { return DJ_; }
	inline const sparse_hessian_type& getHJ() const { return HJ_; }
	inline const tm_base_mat_type& getT() const { return T_; }

    private:
	DATA& data_;

	param_type lambda_, eps2_;
	weights_mat weightsX_, weightsY_;

	tm_base_mat_type T_;
	gradient_type DJ_;
	sparse_hessian_type HJ_;
};


//--------Implementation FIRSTORDER-----/

template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
const FUNCTIONAL_DISC Functional<FIRSTORDER, disc, MANIFOLD, DATA >::disc_type = disc;

// Update the Weights 2D
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA >::updateWeights(){

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t Update Weights..." << std::endl;
    #endif

    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    
    // Subimage boxes
    vpp::box2d without_last_col(vpp::vint2(0,0), vpp::vint2(nr-1, nc-2)); // subdomain without last column
    vpp::box2d without_first_col(vpp::vint2(0,1), vpp::vint2(nr-1, nc-1)); // subdomain without first column
    vpp::box2d without_last_row(vpp::vint2(0,0), vpp::vint2(nr-2, nc-1)); // subdomain without last row
    vpp::box2d without_first_row(vpp::vint2(1,0), vpp::vint2(nr-1, nc-1)); // subdomain without first row

    weightsX_ = weights_mat(data_.img_.domain());
    weightsY_ = weights_mat(data_.img_.domain());

    auto calc_dist = [&] (weights_type& w, const value_type i, const value_type n) {
	w = MANIFOLD::dist_squared(i, n);
    };

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Horizontal neighbours " << std::endl;
    #endif

    // Horizontal Neighbours
    vpp::fill(weightsX_, 0.0);
    vpp::pixel_wise(weightsX_ | without_last_col, data_.img_ | without_last_col, data_.img_ | without_first_col )(/*vpp::_no_threads*/) | calc_dist;

    #ifdef TV_FUNC_DEBUG 
	data_.output_weights(weightsX_,"XWeights.csv");
    #endif	
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Vertical neighbours" << std::endl;
    #endif

    // Vertical Neighbours
    vpp::fill(weightsY_, 0.0);
    vpp::pixel_wise(weightsY_ | without_last_row, data_.img_ | without_last_row, data_.img_ | without_first_row )(/*vpp::_no_threads*/) | calc_dist;

    #ifdef TV_FUNC_DEBUG 
	data_.output_weights(weightsY_,"YWeights.csv");
    #endif	

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Reweighting" << std::endl;
    #endif

    if(disc==ISO){
	auto g =  [&] (const weights_type& ew, weights_type& x, weights_type& y) { x = ew / std::sqrt(x+y+eps2_); y = x; };
	vpp::pixel_wise(data_.edge_weights_, weightsX_, weightsY_) | g ;
    }
    else{
	auto g =  [&] (const weights_type& ew, weights_type& w) { w = ew / std::sqrt(w+eps2_); };
	vpp::pixel_wise(data_.edge_weights_, weightsX_) | g ;
	vpp::pixel_wise(data_.edge_weights_, weightsY_) | g ;
    }
}

// Update the Tangent space ONB
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA >::updateTMBase(){
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tUpdate tangent space basis..." << std::endl;
    #endif


   tm_base_mat_type T(data_.img_.domain());
   vpp::pixel_wise(T, data_.img_)(/*vpp::_no_threads*/) | [&] (tm_base_type& t, const value_type& i) { MANIFOLD::tangent_plane_base(i,t); };
   T_=T;
    
    #ifdef TV_FUNC_DEBUG 
        output_img(T_,"T.csv");
    #endif
}

// Evaluation of J
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
typename Functional<FIRSTORDER, disc, MANIFOLD, DATA >::result_type Functional<FIRSTORDER, disc, MANIFOLD, DATA >::evaluateJ(){

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
	J2 = vpp::sum( vpp::pixel_wise(weightsX_, weightsY_) | [&] (const weights_type& wx, const weights_type& wy) {return 1.0/wx +1.0/wy;} );
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "J1: " << J1 << std::endl;
	std::cout << "J2: " << J2 << std::endl;
    #endif

    return 0.5 * J1 + lambda_* J2;
}

// Evaluation of J'
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA >::evaluateDJ(){

    #ifdef TV_FUNC_WONES_DEBUG 
	output_img(data_.img_,"img.csv");
	vpp::fill(weightsX_, 1.0); // Reset for Debugging
	vpp::fill(weightsY_, 1.0); // Reset for Debugging
    #endif

    img_type grad = img_type(data_.img_.domain());
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tGradient evaluation..." << std::endl;
	std::cout << "\t\t...Fidelity part" << std::endl;
    #endif
    //GRADIENT OF FIDELITY TERM
    if(data_.doInpaint()){
	auto f = [] (value_type& g, const value_type& i, const value_type& n, const bool inp ) { MANIFOLD::deriv1x_dist_squared(i,n,g); g*=(1-inp); };
	vpp::pixel_wise(grad, data_.img_, data_.noise_img_, data_.inp_) | f;
    }
    else{
	auto f = [] (value_type& g, const value_type& i, const value_type& n) { MANIFOLD::deriv1x_dist_squared(i,n,g); };
	vpp::pixel_wise(grad, data_.img_, data_.noise_img_)(/*vpp::_no_threads*/) | f;
    }
    
    //GRADIENT OF TV TERM

    // Subimage boxes
    vpp::box2d without_last_col(vpp::vint2(0,0), vpp::vint2(nr-1, nc-2)); // subdomain without last column
    vpp::box2d without_first_col(vpp::vint2(0,1), vpp::vint2(nr-1, nc-1)); // subdomain without first column
    vpp::box2d without_last_row(vpp::vint2(0,0), vpp::vint2(nr-2, nc-1)); // subdomain without last row
    vpp::box2d without_first_row(vpp::vint2(1,0), vpp::vint2(nr-1, nc-1)); // subdomain without first row

    auto calc_first_arg_deriv = [&] (value_type& x, const weights_type& w, const value_type& i, const value_type& n) { MANIFOLD::deriv1x_dist_squared(i, n, x); x *= w; };
    auto calc_second_arg_deriv = [&] (value_type& y, const weights_type& w, const value_type& i, const value_type& n) { MANIFOLD::deriv1y_dist_squared(i, n, y); y *= w; };
    auto add_to_gradient = [&] (value_type& g, const value_type& d) { g+=d*lambda_; };

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\tGradient evaluation..." << std::endl;
	std::cout << "\t\t...TV part" << std::endl;
	std::cout << "\t\t...-> XD1" << std::endl;
    #endif
    // Horizontal derivatives and weighting
    // ... w.r.t. to first argument
    { // Temporary image XD1 is deallocated after this scope 
	img_type XD1 = img_type(without_last_col);
	vpp::pixel_wise(XD1, weightsX_ | without_last_col, data_.img_ | without_last_col, data_.img_ | without_first_col) | calc_first_arg_deriv;
	#ifdef TV_FUNC_DEBUG 
	    output_img(XD1,"XD1.csv");
	#endif
	auto grad_subX1  = grad | without_last_col;
	vpp::pixel_wise(grad_subX1, XD1) | add_to_gradient;
    } 
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...-> XD2" << std::endl;
    #endif
    // ... w.r.t. second argument
    {
	img_type XD2 = img_type(without_last_col);
	vpp::pixel_wise(XD2, weightsX_ | without_last_col, data_.img_ | without_last_col, data_.img_ | without_first_col) | calc_second_arg_deriv;
	#ifdef TV_FUNC_DEBUG 
	    output_img(XD2,"XD2.csv");
	#endif
	auto grad_subX2  = grad | without_first_col;
	vpp::pixel_wise(grad_subX2, XD2) | add_to_gradient;
    }

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...-> YD1" << std::endl;
    #endif
    // Vertical derivatives and weighting
    // ... w.r.t. first argument
    {
	img_type YD1 = img_type(without_last_row);
	vpp::pixel_wise(YD1, weightsY_ | without_last_row, data_.img_ | without_last_row, data_.img_ | without_first_row) | calc_first_arg_deriv;
	#ifdef TV_FUNC_DEBUG 
	    output_img(YD1,"YD1.csv");
	#endif
	auto grad_subY1  = grad | without_last_row;
	vpp::pixel_wise(grad_subY1, YD1) | add_to_gradient;
    }
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...-> YD2" << std::endl;
    #endif
    // ... w.r.t second argument
    {
	img_type YD2 = img_type(without_last_row);
	vpp::pixel_wise(YD2, weightsY_ | without_last_row, data_.img_ | without_last_row, data_.img_ | without_first_row) | calc_second_arg_deriv;
	#ifdef TV_FUNC_DEBUG 
	    output_img(YD2,"YD2.csv");
	#endif
        auto grad_subY2  = grad | without_first_row;
        vpp::pixel_wise(grad_subY2, YD2) | add_to_gradient;
    }

    
    #ifdef TV_FUNC_DEBUG 
	output_img(grad,"grad.csv");
    #endif

    DJ_ = gradient_type::Zero(nr*nc*manifold_dim); 
    
    // flatten rowwise
    //vpp::pixel_wise(grad, grad.domain()) | [&] (value_type& p, vpp::vint2 coord) { DJ_.segment(3*(nc*coord[0]+coord[1]), value_dim) = p; };
    
    // Apply tangent space restriction and flatten colwise (as in Matlab code)
    updateTMBase();
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Local to global insert" << std::endl;
    #endif
    auto insert2grad = [&] (const tm_base_type& t, const value_type& p, const vpp::vint2 coord) { 
	DJ_.segment(manifold_dim * (coord[0] + nr * coord[1]), manifold_dim) = t.transpose() * Eigen::Map<const Eigen::VectorXd>(p.data(), p.size()); 
	//DJ_.segment(manifold_dim * (coord[0] + nr * coord[1]), manifold_dim) = t.transpose()*p; //remove: does not work for matrix valued pixel
    };

    vpp::pixel_wise(T_, grad, grad.domain()) | insert2grad; 

    #ifdef TV_FUNC_DEBUG 
	std::fstream f;
	f.open("gradJ.csv",std::fstream::out);
	f << DJ_;
	f.close();
    #endif
}

// Evaluation of Hessian J
template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA >::evaluateHJ(){
    #ifdef TV_FUNC_WONES_DEBUG 
	vpp::fill(weightsX_, 1.0); // Reset for Debugging
	vpp::fill(weightsY_, 1.0); // Reset for Debugging
    #endif

    hessian_type hessian(data_.img_.domain());
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    int sparsedim = nr*nc*manifold_dim;
    
        
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
	vpp::pixel_wise(hessian, data_.img_, data_.noise_img_, data_.inp_) | f;
    }
    else{
	auto f = [] (deriv2_type& h, const value_type& i, const value_type& n) { MANIFOLD::deriv2xx_dist_squared(i,n,h); };
	vpp::pixel_wise(hessian, data_.img_, data_.noise_img_) | f;
    }
    
    #ifdef TV_FUNC_DEBUG
	output_matval_img(hessian,"HF0.csv");
    #endif

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Local to global insert" << std::endl;
    #endif
   //TODO: Check whether all 2nd-derivative matrices are symmetric s.t. only half the matrix need to be traversed. e.g. local_col=local_row instead of 0
    auto local2globalInsert = [&](const tm_base_type& t, const deriv2_type& h, const vpp::vint2 coord) { 
	int pos = manifold_dim * (coord[0] + nr * coord[1]); // columnwise flattening
	restricted_deriv2_type ht=t.transpose()*h*t;
	for(int local_row = 0; local_row<ht.rows(); local_row++)
	    for(int local_col = local_row; local_col < ht.cols(); local_col++){
		scalar_type e = ht(local_row, local_col);
		if(e!=0){
		    int global_row = pos + local_row;
		    int global_col = pos + local_col;
		    triplist.push_back(Trip(global_row,global_col,e));
		    if(global_row != global_col)
			triplist.push_back(Trip(global_col,global_row,e));
		    //HF.insert(global_row, global_col) = e;
		    //if(global_row != global_col)
		    //	HF.insert(global_col, global_row) = e;
		    }
	    }
    };

    vpp::pixel_wise(T_, hessian, hessian.domain())(vpp::_no_threads) | local2globalInsert;
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
    triplist.reserve(3*sparsedim*manifold_dim);

    // Subimage boxes
    vpp::box2d without_last_col(vpp::vint2(0,0), vpp::vint2(nr-1, nc-2)); // subdomain without last column
    vpp::box2d without_first_col(vpp::vint2(0,1), vpp::vint2(nr-1, nc-1)); // subdomain without first column
    vpp::box2d without_last_row(vpp::vint2(0,0), vpp::vint2(nr-2, nc-1)); // subdomain without last row
    vpp::box2d without_first_row(vpp::vint2(1,0), vpp::vint2(nr-1, nc-1)); // subdomain without first row

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
	hessian_type XD11(without_last_col);
        vpp::pixel_wise(XD11, weightsX_ | without_last_col, data_.img_ | without_last_col, data_.img_ | without_first_col) | calc_xx_der;
        #ifdef TV_FUNC_DEBUG 
		output_matval_img(XD11,"XD11.csv");
        #endif
	auto hess_subX11  = hessian | without_last_col;
	vpp::pixel_wise(hess_subX11, XD11) | [&] (deriv2_type& h, const deriv2_type& d) { h=d; };
    }
	#pragma omp parallel for
	for(int r=0; r< nr; r++) 
	    hessian(r,nc-1)=deriv2_type::Zero(); // set last column to zero

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->XD22" << std::endl;
    #endif
    //... w.r.t. second arguments
    {
	hessian_type XD22(without_last_col);
        vpp::pixel_wise(XD22, weightsX_ | without_last_col, data_.img_ | without_last_col, data_.img_ | without_first_col) | calc_yy_der;
	#ifdef TV_FUNC_DEBUG 
	    output_matval_img(XD22,"XD22.csv");
	#endif
	auto hess_subX22  = hessian | without_first_col;
	vpp::pixel_wise(hess_subX22, XD22) | add_to_hessian;
    }

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->YD11" << std::endl;
    #endif
    // Vertical Second Derivatives weighting
    //... w.r.t. first arguments
    {
	hessian_type YD11(without_last_row);
        vpp::pixel_wise(YD11, weightsY_ | without_last_row, data_.img_ | without_last_row, data_.img_ | without_first_row) | calc_xx_der;
	#ifdef TV_FUNC_DEBUG 
	    output_matval_img(YD11,"YD11.csv");
	#endif
	auto hess_subY11  = hessian | without_last_row;
	vpp::pixel_wise(hess_subY11, YD11) | add_to_hessian;
    }

    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->YD22" << std::endl;
    #endif
    //... w.r.t. second arguments
    {
	hessian_type YD22(without_last_row);
        vpp::pixel_wise(YD22, weightsY_ | without_last_row, data_.img_ | without_last_row, data_.img_ | without_first_row) | calc_yy_der;
	#ifdef TV_FUNC_DEBUG 
	    output_matval_img(YD22,"YD22.csv");
	#endif
	auto hess_subY22  = hessian | without_first_row;
	vpp::pixel_wise(hess_subY22, YD22) | [&] (deriv2_type& h, const deriv2_type& d) { h+=d; };
    }
    
    #ifdef TV_FUNC_DEBUG
        output_matval_img(T_, "T.csv");
	output_matval_img(hessian, "NonMixedHessian.csv");
    #endif
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...Local to global insert" << std::endl;
    #endif
    // Insert elementwise into sparse Hessian
    // NOTE: Eventually make single version for both cases, including an offset
    // --> additional parameters sparse_mat, offset
    int row_offset=0;
    int col_offset=0;
    auto local2globalInsertHTV = [&](const tm_base_type& t1, const tm_base_type& t2, const deriv2_type& h, const vpp::vint2 coord) { 
	int pos = manifold_dim*(coord[0]+nr*coord[1]); // columnwise flattening
	restricted_deriv2_type ht = t1.transpose()*h*t2;
	     #ifdef TV_FUNC_DEBUG_VERBOSE2
		std::cout << "\n\nPos: " << pos << " coord[0]=" << coord[0] << " coord[1]=" << coord[1] << std::endl;
		std::cout << "\t\tT_left:" << std::endl;
		std::cout << t1 << std::endl;
		std::cout << "\t\tHbig:" << std::endl;
		std::cout << h << std::endl;
		std::cout << "\t\tT_right:" << std::endl;
		std::cout << t2 << std::endl;
		std::cout << "\t\tHsmall:" << std::endl;
		std::cout << ht << std::endl;
	    #endif
	for(int local_row=0; local_row<ht.rows(); local_row++)
	    for(int local_col=0; local_col<ht.cols(); local_col++){
		scalar_type e = ht(local_row, local_col);
		if(e!=0){
		    int global_row = pos + row_offset + local_row;
		    int global_col = pos + col_offset + local_col;
		    triplist.push_back(Trip(global_row,global_col,e));
		    if(row_offset > 0 || col_offset > 0)
			triplist.push_back(Trip(global_col,global_row,e));
		    //HTV.insert(global_row, global_col) = e;
		    //if(global_col != global_row)
		    //	HTV.insert(global_col, global_row)  = e;
		}
	    }
    };
    vpp::pixel_wise(T_, T_, hessian, hessian.domain())(vpp::_no_threads) | local2globalInsertHTV;
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->XD12" << std::endl;
    #endif              
    // Horizontal Second Derivatives and weighting
    // ... w.r.t. first and second arguments 
    {
	hessian_type XD12(without_last_col);
	vpp::pixel_wise(XD12, weightsX_ | without_last_col, data_.img_ | without_last_col, data_.img_ | without_first_col ) | calc_xy_der;
	#ifdef TV_FUNC_DEBUG 
	    output_matval_img(XD12,"XD12.csv");
	#endif
	#ifdef TV_FUNC_DEBUG_VERBOSE
		std::cout << "\t\t...Local to global insert:" << std::endl;
	#endif
	// Offsets for upper nyth subdiagonal
	row_offset=0;
	col_offset=manifold_dim*nr;
	vpp::pixel_wise(T_ | without_last_col, T_ | without_first_col, XD12, XD12.domain())(vpp::_no_threads) | local2globalInsertHTV;
    }
    
    #ifdef TV_FUNC_DEBUG_VERBOSE
	std::cout << "\t\t...->YD12" << std::endl;
    #endif
    // Vertical Second Derivatives and weighting
    //... w.r.t. second arguments
    {
	hessian_type YD12(data_.img_.domain());
	vpp::pixel_wise(YD12 | without_last_row, weightsY_ | without_last_row, data_.img_ | without_last_row, data_.img_ | without_first_row) | calc_xy_der;
	#ifdef TV_FUNC_DEBUG 
	    output_matval_img(YD12,"YD12.csv");
        #endif
	
	//Set last row to zero
	deriv2_type *lastrow = &YD12(nr-1,0);
	for(int c=0; c< nc; c++) 
	    lastrow[c]=deriv2_type::Zero();

	#ifdef TV_FUNC_DEBUG_VERBOSE
		std::cout << "\t\t...Local to global insert:" << std::endl;
	#endif
	// Offsets for first upper subdiagonal
	row_offset=0;
	col_offset=manifold_dim;
	vpp::pixel_wise(T_ | without_last_row, T_ | without_first_row, YD12 | without_last_row, without_last_row)(vpp::_no_threads) | local2globalInsertHTV;
	
	//Manually insert last row
	for(int c=0; c<nc-1; c++) 
	    local2globalInsertHTV(T_(nr-1,c), T_(nr-1,c + 1), YD12(nr-1,c), vpp::vint2(nr-1, c));
	    //local2globalInsertHTV(firstrow[c], lastrow[c], vpp::vint2(nr-1,c));
	
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
	    if(sparsedim<70){
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
/*	#ifdef TV_FUNC_DEBUG_VERBOSE
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
	f.close();*/
    #endif

}


template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
template < class IMG >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA >::output_img(const IMG& img, const char* filename) const{
    int nr = img.nrows();
    int nc = img.ncols();

    std::fstream f;
    f.open(filename, std::fstream::out);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    for (int r=0; r<nr; r++){
	const auto* cur = &img(r,0);
	for (int c=0; c<nc; c++){
	    f << cur[c].format(CommaInitFmt);
	    if(c != nc-1) f << ",";
	}
	f <<  std::endl;
    }
    f.close();
}

template <enum FUNCTIONAL_DISC disc, typename MANIFOLD, class DATA >
template < class IMG >
void Functional<FIRSTORDER, disc, MANIFOLD, DATA >::output_matval_img(const IMG& img, const char* filename) const{
    int nr = img.nrows();
    int nc = img.ncols();

    std::fstream f;
    f.open(filename, std::fstream::out);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "\n");
    for (int r=0; r<nr; r++){
	const auto* cur = &img(r,0);
	for (int c=0; c<nc; c++)
	    f << cur[c].format(CommaInitFmt);
    }
    f.close();
}

}// end namespace tvtml

#endif
