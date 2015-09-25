#ifndef TVMTL_TVMINIMIZER_PRPT3D_HPP
#define TVMTL_TVMINIMIZER_PRPT3D_HPP

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
    class TV_Minimizer< PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR, 3> {
    
	public:
	    // Manifold typedefs
	    typedef typename MANIFOLD::scalar_type scalar_type;
	    typedef typename MANIFOLD::value_type value_type;

	    // Functional typedefs
	    typedef typename FUNCTIONAL::weights_mat weights_mat;
	    typedef typename FUNCTIONAL::weights_type weights_type;

	    // Data acccess typedefs
	    typedef typename DATA::storage_type img_type;


	    // Constructor
	    TV_Minimizer(FUNCTIONAL& func, DATA& dat):
		func_(func),
		data_(dat)
	    {
		static_assert(FUNCTIONAL::disc_type == ANISO, "Proximal point is only possible with anisotropic weights!");
		static_assert(DATA::img_dim == 3, "Dimension of Proximal point minimizer(DIM=3) and data class must match! ");

		for(int i = 0; i < 7; i++){
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
	
	    img_type proximal_mappings_[7];

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
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::updateFidelity(double muk){
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Fidelity Update with muk: " << muk << std::endl;
    #endif
    
    double tau = muk / (muk + 1.0);
   
    if(data_.doInpaint()){
	auto proximal_mapF = [&] (value_type& p, const value_type& i, const value_type& n, const bool inp ) { 
	    value_type temp;
	    MANIFOLD::convex_combination(i, n, tau, temp);
	    p = temp * (1-inp) + i * inp; 
	};

	pixel_wise3d(proximal_mapF, proximal_mappings_[0], data_.img_, data_.noise_img_, data_.inp_);
    }
    else{
    	auto proximal_mapF = [&] (value_type& p, const value_type& i, const value_type& n) { 
	    MANIFOLD::convex_combination(i, n, tau, p);
	};

	pixel_wise3d(proximal_mapF, proximal_mappings_[0], data_.img_, data_.noise_img_); 
    }

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::updateTV(double muk, int dim, const weights_mat& W){
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t TV Update with muk: " << muk << std::endl;
	    std::cout << "\t\t... Dimension: " << dim << std::endl;
    #endif 
    double lmuk = muk * func_.getlambda();
    weights_mat tau(data_.img_.domain());
    pixel_wise3d([&] (weights_type& t, const weights_type& w) { t = std::min(lmuk * w, 0.5);}, tau, W);

    // 3D
    int ns = data_.img_.nslices();
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    
    // Subimage boxes
    vpp::box3d without_last_x(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 1, nc - 2)); // subdomain without last xslice
    vpp::box3d without_last_y(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 2, nc - 1)); // subdomain without last yslice
    vpp::box3d without_last_z(vpp::vint3(0,0,0), vpp::vint3(ns - 2, nr - 1, nc - 1)); // subdomain without last zslice
    vpp::box3d without_first_x(vpp::vint3(0,0,1), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first xlice
    vpp::box3d without_first_y(vpp::vint3(0,1,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first yslice
    vpp::box3d without_first_z(vpp::vint3(1,0,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first zslice

    // Dimensions of slice
    vpp::vint3 dims(ns, nr, nc); dims(dim) = 2; // z-dim=0, y-dim = 1, x-dim = 2
    vpp::vint3 loopdims(ns, nr, nc); loopdims(dim) = 1;

    // Relative neighbor coordinates
    vpp::vint3 n(0,0,0); n(dim) = 1;  
    
    // Correct Subimage
    vpp::box3d subimage = data_.img_.domain();
    if(dim == 0  && (ns % 2 != 0)) // dim = z
	    subimage = without_last_z; 

    if(dim == 1  && (nr % 2 != 0)) // dim = y
	    subimage = without_last_y; 

    if(dim == 2  &&  (nc % 2 != 0)) // dim = x
	    subimage = without_last_x;


    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t\t...Odd Pairs..., dims = (" << dims(0) << "," << dims(1) << "," << dims(2) <<  "), n = (" << n(0) << "," << n(1) << "," << n(2) << ")" <<std::endl;
	    int num_blocks = 0;
    #endif 
    
    clone3d(data_.img_, proximal_mappings_[1 + 2 * dim]);
    auto I = data_.img_ | subimage;
    auto T = tau | subimage;
    auto P = proximal_mappings_[1 + 2 * dim] | subimage;
    
    int Ins = I.nslices();
    int Inr = I.nrows();
    int Inc = I.ncols();
    
    for(int s = 0; s < Ins; s += dims(0)){
	//#pragma omp parallel for
	for(int r = 0; r < Inr; r += dims(1)){
	    for(int c = 0; c < Inc; c+= dims(2)){
		vpp::vint3 first(s,r,c);
		vpp::vint3 last(s + loopdims(0) - 1, r + loopdims(1) - 1, c + loopdims(2)-1 );
		vpp::box3d block(first, last);
		    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
			std::cout << "\t\t\tBlock: " << num_blocks << " , Slice: " << s << ", Row: " << r << ", Col: " << c << std::endl;
		    #endif
	
		for (auto p : block){
		    value_type l;
		    MANIFOLD::log(I(p), I(p + n), l);
		    MANIFOLD::exp(I(p), l * T(p), P(p));
		    MANIFOLD::exp(I(p), l - l * T(p), P(p + n));
		}
	    	#ifdef TVMTL_TVMIN_DEBUG
		    num_blocks++;
		#endif
	    }
	}
    }

    #ifdef TVMTL_TVMIN_DEBUG
	std::cout << "\t\tBlock processed: " << num_blocks << std::endl;
    #endif

   if(dim == 0) // z
	if(ns % 2 == 0)
	    subimage =  vpp::box3d(vpp::vint3(1, 0, 0), vpp::vint3(ns-2, nr-1, nc-1)); // without first and last z slice
	else
	    subimage = without_first_z; 
 
    if(dim == 1) // y
	if(nr % 2 == 0)
	    subimage =  vpp::box3d(vpp::vint3(0, 1, 0), vpp::vint3(ns-1, nr-2, nc-1)); 
	else
	    subimage = without_first_y; 

    if(dim == 2) // x 
	if(nc % 2 == 0)
	    subimage = vpp::box3d(vpp::vint3(0, 0, 1), vpp::vint3(ns-1, nr-1, nc-2)); 
	else
	    subimage = without_first_x;  

    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t\t...Even Pairs..." << std::endl;
	    num_blocks = 0;
    #endif 
    clone3d(data_.img_, proximal_mappings_[2 + 2 * dim]);
    I = data_.img_ | subimage;
    T = tau | subimage;
    P = proximal_mappings_[2 + 2 * dim] | subimage;

    Ins = I.nslices();
    Inr = I.nrows();
    Inc = I.ncols();
    

    for(int s = 0; s < Ins; s += dims(0)){
	//#pragma omp parallel for
	for(int r = 0; r < Inr; r += dims(1)){
	    for(int c = 0; c < Inc; c+= dims(2)){
		vpp::vint3 first(s,r,c);
		vpp::vint3 last(s + loopdims(0) - 1, r + loopdims(1) - 1, c + loopdims(2)-1 );
		vpp::box3d block(first, last);
		    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
			std::cout << "\t\t\tBlock: " << num_blocks << " , Slice: " << s << ", Row: " << r << ", Col: " << c << std::endl;
		    #endif
	
		for (auto p : block){
		    value_type l;
		    MANIFOLD::log(I(p), I(p + n), l);
		    MANIFOLD::exp(I(p), l * T(p), P(p));
		    MANIFOLD::exp(I(p), l - l * T(p), P(p + n));
		}
	    	#ifdef TVMTL_TVMIN_DEBUG
		    num_blocks++;
		#endif
	    }
	}
    }

    #ifdef TVMTL_TVMIN_DEBUG
	std::cout << "\t\tBlocks processed: " << num_blocks << std::endl;
    #endif
 
}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::geod_mean(){
    typedef value_type& vtr;
    typedef const vtr cvtr;
    
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Calculate Geodesic mean: "<< std::endl;
	    std::cout << "\t\t...first guess euclidiean mean: "<< std::endl;
    #endif 

    // First estimate
    auto euclidian_mean = [&] (vtr i, cvtr p0, cvtr p1, cvtr p2, cvtr p3, cvtr p4, cvtr p5, cvtr p6) {
	i = (p0 + p1 + p2 + p3 + p4 + p5 + p6) / 7.0;
    };
    pixel_wise3d(euclidian_mean, data_.img_, proximal_mappings_[0], proximal_mappings_[1], proximal_mappings_[2], proximal_mappings_[3], proximal_mappings_[4], proximal_mappings_[5], proximal_mappings_[6]);

    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t\t...Karcher mean Iteration: "<< std::endl;
    #endif 

    //TODO: Put in Algo traits
    double tol = 1e-10;
    double max_karcher_iterations = 15;

    auto karcher_mean = [&] (vtr i, cvtr p0, cvtr p1, cvtr p2, cvtr p3, cvtr p4, cvtr p5, cvtr p6) {
    
	/* Slow version, - needs to copy p_i for insertion in value_list
	typename MANIFOLD::value_list v;
	v.push_back(p0);
	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	v.push_back(p5);
	v.push_back(p6);
	MANIFOLD::karcher_mean(i, v, tol, max_karcher_iterations); */
    
	MANIFOLD::karcher_mean(i, p0, p1, p2, p3, p4, p5, p6);
    };

    pixel_wise3d(karcher_mean, data_.img_, proximal_mappings_[0], proximal_mappings_[1], proximal_mappings_[2], proximal_mappings_[3], proximal_mappings_[4], proximal_mappings_[5], proximal_mappings_[6]);

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::approx_mean(){
    typedef value_type& vtr;
    typedef const vtr cvtr;
    
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Calculate approximate mean via convex combinations: "<< std::endl;
    #endif 
    //TODO: Check weights
    auto convex_comb_mean = [&] (vtr i, cvtr p0, cvtr p1, cvtr p2, cvtr p3, cvtr p4, cvtr p5, cvtr p6) {
	value_type p01, p23, p45;
	MANIFOLD::convex_combination(p0, p1, 0.5, p01);
	MANIFOLD::convex_combination(p2, p3, 0.5, p23);
	MANIFOLD::convex_combination(p4, p5, 0.5, p45);
	MANIFOLD::convex_combination(p01, p23, 0.5, p01);
	MANIFOLD::convex_combination(p45, p6, 1.0/3, p23);
	MANIFOLD::convex_combination(p01, p23, 3.0/7.0, i);
    };

    pixel_wise3d(convex_comb_mean, data_.img_, proximal_mappings_[0], proximal_mappings_[1], proximal_mappings_[2], proximal_mappings_[3], proximal_mappings_[4], proximal_mappings_[5], proximal_mappings_[6]);

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::prpt_step(double muk){
    #ifdef TVMTL_TVMIN_DEBUG
	    std::cout << "\t Proximal step with muk: " << muk << std::endl;
    #endif
    weights_mat X = func_.getweightsX();
    weights_mat Y = func_.getweightsY();
    weights_mat Z = func_.getweightsZ();

    updateFidelity(muk);
    
    updateTV(muk, 0, Z);
    updateTV(muk, 1, Y);
    updateTV(muk, 2, X);
    
    if(use_approximate_mean_)
	approx_mean();
    else
	geod_mean();
}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::minimize(){
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
