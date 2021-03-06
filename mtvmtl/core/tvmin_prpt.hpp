#ifndef TVMTL_TVMINIMIZER_PRPT_HPP
#define TVMTL_TVMINIMIZER_PRPT_HPP

//System includes
#include <iostream>
#include <map>
#include <vector>
#include <chrono>

#ifdef TVMTL_TVMIN_DEBUG
    #include <string>
#endif

//Eigen includes
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>

//CGAL includes For linear Interpolation
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>


//vpp includes
#include <vpp/vpp.hh>


namespace tvmtl {

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
    class TV_Minimizer< PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR > {
    
	public:
	    // Manifold typedefs
	    typedef typename MANIFOLD::scalar_type scalar_type;
	    typedef typename MANIFOLD::value_type value_type;

	    // Functional typedefs
	    typedef typename FUNCTIONAL::weights_mat weights_mat;
	    typedef typename FUNCTIONAL::weights_type weights_type;

	    // Data acccess typedefs
	    typedef vpp::box_nbh2d<value_type,3,3> nbh_type;
	    typedef typename DATA::storage_type img_type;


	    // Constructor
	    TV_Minimizer(FUNCTIONAL& func, DATA& dat):
		func_(func),
		data_(dat)
	    {
		static_assert(FUNCTIONAL::disc_type == ANISO, "Proximal point is only possible with anisotropic weights!");

		for(int i = 0; i < 5; i++){
		    proximal_mappings_[i] = img_type(data_.img_.domain());
		}
		use_approximate_mean_ = false;
		max_prpt_steps_ = 50;
		tolerance_ = 1e-4;
		max_runtime_ = 1000.0;
	    }
	    
	    void use_approximate_mean(bool u) { use_approximate_mean_ = u; }

	    void first_guess();
	    
	    void updateFidelity(double muk);
	    void updateTV(double muk, int dim, const weights_mat& W);

	    void geod_mean();
	    void approx_mean();

	    void prpt_step(double muk);
	    void minimize();

	    void setMax_runtime(int t) { max_runtime_ = t; }
	    void setMax_prpt_steps(int n) { max_prpt_steps_ = n; }
	    void setTolerance(double t) {tolerance_ =t; }
	    
	    int max_runtime(int t) const { return max_runtime_; }
	    int max_prpt_steps(int n) const { return max_prpt_steps_; }
	    int tolerance(double t) const { return tolerance_; }
	

	private:
	    FUNCTIONAL& func_;
	    DATA& data_;
	
	    img_type proximal_mappings_[5];

	    bool use_approximate_mean_;

	    int prpt_step_;
	    int max_prpt_steps_;
	    double tolerance_;
	    double max_runtime_;

	    std::vector< std::chrono::duration<double> > Ts_;
	    std::vector< typename FUNCTIONAL::result_type > Js_;
    };

/*----- IMPLEMENTATION PRPT------*/
template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::updateFidelity(double muk){
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Fidelity Update with muk: " << muk << std::endl;
	    func_.output_img(data_.img_, "prptImg.csv");
	    func_.output_img(data_.noise_img_,"prptNoiseImg.csv");
	    func_.output_img(proximal_mappings_[0],"prptProxMap.csv");
    #endif
    
    double tau = muk / (muk + 1.0);
   
    if(data_.doInpaint()){
	auto proximal_mapF = [&] (value_type& p, const value_type& i, const value_type& n, const bool inp ) { 
	    value_type temp;
	    MANIFOLD::convex_combination(i, n, tau, temp);
	    p = temp * (1-inp) + i * inp; 
	};

	vpp::pixel_wise(proximal_mappings_[0], data_.img_, data_.noise_img_, data_.inp_) | proximal_mapF; 
    }
    else{
    	auto proximal_mapF = [&] (value_type& p, const value_type& i, const value_type& n) { 
	    MANIFOLD::convex_combination(i, n, tau, p);
	};

	vpp::pixel_wise(proximal_mappings_[0], data_.img_, data_.noise_img_) | proximal_mapF; 
    }

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::updateTV(double muk, int dim, const weights_mat& W){
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t TV Update with muk: " << muk << std::endl;
	    std::cout << "\t\t... Dimension: " << dim << std::endl;
    #endif 
    double lmuk = muk * func_.getlambda();
    weights_mat tau = vpp::pixel_wise(W) | [&] (const weights_type& w) {return std::min(lmuk * w, 0.5); }; 

    // 2D
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();

    // Dimensions of slice
    vpp::vint2 dims(nr, nc); dims(dim) = 2; // y-dim = 0, x-dim = 1
    vpp::vint2 loopdims(nr, nc); loopdims(dim) = 1; // y-dim = 0, x-dim = 1
    //vpp::vint3 b(nz, ny, nx); dims(dim) =2; 

    // Relative neighbor coordinates
    vpp::vint2 n(0,0); n(dim) = 1;  
    
    // Correct Subimage
    vpp::box2d subimage = data_.img_.domain();
    if(dim == 0  && (nr % 2 != 0)) // dim = y
	    subimage =  vpp::box2d(vpp::vint2(0,0), vpp::vint2(nr-2, nc-1)); // subdomain without last row

    if(dim == 1  &&  (nc % 2 != 0)) // dim =x
	    subimage = vpp::box2d(vpp::vint2(0,0), vpp::vint2(nr-1, nc-2)); // subdomain without last columns


    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t\t...Odd Pairs..., dims = (" << dims(0) << "," << dims(1) << "), n = (" << n(0) << "," << n(1) << ")" <<std::endl;
	    int num_blocks = 0;
    #endif 
    
    auto blockwise_proximal_map = [&] (const auto I, const auto T, auto P) {
	for(int r = 0; r < loopdims(0); ++r ){
	    for(int c = 0; c < loopdims(1); ++c){
	    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
		    std::cout << "\t\t\tBlock: " << num_blocks << ", Row: " << r << ", Col: " << c << std::endl;
	    #endif
		value_type l;
		MANIFOLD::log(I(r,c), I(vpp::vint2(r,c) + n), l);
		MANIFOLD::exp(I(r,c), l * T(r,c), P(r,c));
		MANIFOLD::exp(I(r,c), l - l * T(r,c), P(vpp::vint2(r,c) + n));
	    }
	}
    #ifdef TVMTL_TVMIN_DEBUG
	num_blocks++;
    #endif
    };

    proximal_mappings_[1 + 2 * dim] = vpp::clone(data_.img_);
    vpp::block_wise(dims, data_.img_ | subimage, tau | subimage, proximal_mappings_[1 + 2 * dim] | subimage) | blockwise_proximal_map;
    #ifdef TVMTL_TVMIN_DEBUG
	std::cout << "\t\tBlock processed: " << num_blocks << std::endl;
    #endif
    
    if(dim == 0 /* y */)
	if(nr % 2 == 0)
	    subimage =  vpp::box2d(vpp::vint2(1,0), vpp::vint2(nr-2, nc-1)); // subdomain without first and last row
	else
	    subimage =  vpp::box2d(vpp::vint2(1,0), vpp::vint2(nr-1, nc-1)); // subdomain without first row


    if(dim == 1 /* x */)
	if(nc % 2 == 0)
	    subimage = vpp::box2d(vpp::vint2(0,1), vpp::vint2(nr-1, nc-2)); // subdomain without first and last column
	else
	    subimage = vpp::box2d(vpp::vint2(0,1), vpp::vint2(nr-1, nc-1)); // subdomain without first column

    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t\t...Even Pairs..." << std::endl;
	    num_blocks = 0;
    #endif 
    proximal_mappings_[2 + 2 * dim] = vpp::clone(data_.img_);
    vpp::block_wise(dims, data_.img_ | subimage, tau | subimage, proximal_mappings_[2 + 2 * dim] | subimage) | blockwise_proximal_map;
	
    #ifdef TVMTL_TVMIN_DEBUG
	std::cout << "\t\tBlocks processed: " << num_blocks << std::endl;
    #endif
 
}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::geod_mean(){
    typedef value_type& vtr;
    typedef const vtr cvtr;
    
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Calculate Geodesic mean: "<< std::endl;
	    std::cout << "\t\t...first guess euclidiean mean: "<< std::endl;
    #endif 

    // First estimate
    auto euclidian_mean = [&] (vtr i, cvtr p0, cvtr p1, cvtr p2, cvtr p3, cvtr p4) {
	i = (p0 + p1 + p2 + p3 + p4) / 5.0;
    };
    vpp::pixel_wise(data_.img_, proximal_mappings_[0], proximal_mappings_[1], proximal_mappings_[2], proximal_mappings_[3], proximal_mappings_[4]) | euclidian_mean;

    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t\t...Karcher mean Iteration: "<< std::endl;
    #endif 

    //TODO: Put in Algo traits
    double tol = 1e-10;
    double max_karcher_iterations = 15;

    auto karcher_mean = [&] (vtr i, cvtr p0, cvtr p1, cvtr p2, cvtr p3, cvtr p4) {
    
	/* Slow version, - needs to copy p_i for insertion in value_list
	typename MANIFOLD::value_list v;
	v.push_back(p0);
	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	MANIFOLD::karcher_mean(i, v, tol, max_karcher_iterations); */
    
	MANIFOLD::karcher_mean(i, p0, p1, p2, p3, p4);
    };

    vpp::pixel_wise(data_.img_, proximal_mappings_[0], proximal_mappings_[1], proximal_mappings_[2], proximal_mappings_[3], proximal_mappings_[4]) | karcher_mean;

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::approx_mean(){
    typedef value_type& vtr;
    typedef const vtr cvtr;
    
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Calculate approximate mean via convex combinations: "<< std::endl;
    #endif 

    auto convex_comb_mean = [&] (vtr i, cvtr p0, cvtr p1, cvtr p2, cvtr p3, cvtr p4) {
	value_type p01, p23;
	MANIFOLD::convex_combination(p0, p1, 0.5, p01);
	MANIFOLD::convex_combination(p2, p3, 0.5, p23);
	MANIFOLD::convex_combination(p01, p23, 0.5, i);
	MANIFOLD::convex_combination(i, p4, 0.2, i);
    };

    vpp::pixel_wise(data_.img_, proximal_mappings_[0], proximal_mappings_[1], proximal_mappings_[2], proximal_mappings_[3], proximal_mappings_[4]) | convex_comb_mean;

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::prpt_step(double muk){
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Proximal step with muk: " << muk << std::endl;
    #endif
    weights_mat X = func_.getweightsX();
    weights_mat Y = func_.getweightsY();

    updateFidelity(muk);
    
    updateTV(muk, 0, Y);
    updateTV(muk, 1, X);
    
    if(use_approximate_mean_)
	approx_mean();
    else
	geod_mean();
}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::minimize(){
    std::cout << "Starting Proximal Point Algorithm with..." << std::endl;
    std::cout << "\t Lambda = \t" << func_.getlambda() << std::endl;
    std::cout << "\t Max Steps = \t" << max_prpt_steps_ << std::endl;
    
    
    prpt_step_ = 1;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> t = std::chrono::duration<double>::zero();
    start = std::chrono::system_clock::now();
    
    // PRPT Iteration Loop
    while(prpt_step_ <= max_prpt_steps_ && t.count() < max_runtime_){
	
	std::cout << "PRPT Step #" << prpt_step_ << std::endl;
	typename FUNCTIONAL::result_type J = func_.evaluateJ();
	Js_.push_back(J);
	std::cout << "\t Value of Functional J: " << J << std::endl;
	
	double muk = 3.0 * std::pow(static_cast<double>(prpt_step_),-0.95);
	#ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Value of muk: " << muk << std::endl;
	#endif

	prpt_step(muk);
	
	end = std::chrono::system_clock::now();
	t = end - start; 
	std::cout << "\t Elapsed time: " << t.count() << " seconds." << std::endl;
	prpt_step_++;
	Ts_.push_back(t);
    }

    std::cout << "Minimization in " << t.count() << " seconds." << std::endl;
}



} // end namespace tvmtl


#endif
