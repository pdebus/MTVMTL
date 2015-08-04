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
	    typedef typename DATA::img_type img_type;
	    
	    // Constructor
	    TV_Minimizer(FUNCTIONAL& func, DATA& dat):
		func_(func),
		data_(dat)
	    {
		for(int i=0; i<5; i++)
		    proximal_mappings_ = img_type(data_.img_.domain());
	    }

	    void first_guess();
	    
	    void updateFidelity(double muk);
	    void updateTV(double muk, int dim, const weights_mat& W);
	    void geod_mean();

	    void prpt_step(double muk);
	    void minimize();

	    void output() { std::cout << "OUTPUT TEST" << std::endl; }
	

	private:
	    FUNCTIONAL& func_;
	    DATA& data_;
	
	    img_type proximal_mappings_[5];

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
    double tau = muk / (muk + 1.0);
    
    auto proximal_mapF = [&] (value_type& p, const value_type& i, const value_type& n, const bool inp ) { 
	value_type l, e;
	MANIFOLD::log(i, n, l);
	MANIFOLD::exp(i, l*tau, e);
	return e * (1-inp) + i * inp; 
    };

    vpp::pixel_wise(proximal_mappings_[0], data_.img_, data_.noise_img_, data_.inp_) | proximal_mapF; 
}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::updateTV(double muk, int dim, const weights_mat& W){
    
    double lmuk = muk * func_.getlambda();
    weights_mat tau = vpp::pixel_wise(W) | [&] (const weights_type& w) {return std::min(lmuk * w, 0.5); }; 

    // 2D
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();

    // Dimensions of slice
    vpp::vint2 dims(nr, nc); dims(dim) = 2; // y-dim = 0, x-dim = 1
    //vpp::vint3 b(nz, ny, nx); dims(dim) =2; 

    // Relative neighbor coordinates
    vpp::vint2 n(0,0); n(dim) = 1;  
    
    // Correct Subimage
    vpp::box2d subimage;

    if(dim == 0  && (nr % 2 != 0)) // dim = y
	    subimage =  vpp::box2d(vpp::vint2(0,0), vpp::vint2(nr-2, nc-1)); // subdomain without last row

    if(dim == 1  &&  (nc % 2 != 0)) // dim =x
	    subimage = vpp::box2d(vpp::vint2(0,0), vpp::vint2(nr-1, nc-2)); // subdomain without last columns

    

    proximal_mappings_[1 + 2 * dim] = vpp::clone(data_.img_);
    vpp::block_wise(dims, data_.img_ | subimage, tau | subimage, proximal_mappings_[1 + 2 * dim] | subimage) | [&] (const auto& I, const auto& T, auto& P) {
	for(int r = 0; r < dims(0); ++r ){
	    for(int c = 0; c < dims(1); ++c){
		value_type l;
		MANIFOLD::log(I(r,c), I(vpp::vint2(r,c) + n), l);
		MANIFOLD::exp(I(r,c), l * T(r,c), P(r,c));
		MANIFOLD::exp(I(r,c), l - l * T(r,c), P(vpp::vint2(r,c) + n));
	    }
	}
    };
    
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


    proximal_mappings_[2 + 2 * dim] = vpp::clone(data_.img_);
    vpp::block_wise(dims, data_.img_ | subimage, tau | subimage, proximal_mappings_[2 + 2 * dim] | subimage) | [&] (const auto& I, const auto& T, auto& P) {
	for(int r = 0; r < dims(0); ++r ){
	    for(int c = 0; c < dims(1); ++c){
		value_type l;
		MANIFOLD::log(I(vpp::vint2(r,c) + n), I(r,c), l);
		MANIFOLD::exp(l * T(r,c), I(r,c), P(r,c));
		MANIFOLD::exp(l - l * T(r,c), I(r,c), P(vpp::vint2(r,c) + n));
	    }
	}
    };
 
}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::geod_mean(){
    typedef value_type& vtr;
    typedef const vtr cvtr;
    
    // First estimate
    auto euclidian_mean = [&] (vtr i, cvtr p0, cvtr p1, cvtr p2, cvtr p3, cvtr p4) {
	i = (p0 + p1 + p2 + p3 + p4) / 5.0;
    };

    vpp::pixel_wise(data_.img_, proximal_mappings_[0], proximal_mappings_[1], proximal_mappings_[2], proximal_mappings_[3], proximal_mappings_[4]) | euclidian_mean;
    auto karcher_mean = [&] (vtr i, cvtr p0, cvtr p1, cvtr p2, cvtr p3, cvtr p4) {
	value_type l0, l1, l2, l3, l4;
	MANIFOLD::log(i, p0, l0);
	MANIFOLD::log(i, p1, l1);
	MANIFOLD::log(i, p2, l2);
	MANIFOLD::log(i, p3, l3);
	MANIFOLD::log(i, p4, l4);
	MANIFOLD::exp(i, (l0 + l1 + l2 + l3 + l4) / 5.0, i);
    };

    weights_mat diff(data_.img_.domain());

    //TODO: Put in Algo traits
    double tol = 1e-11;
    double error = 1.0;

    while(error > tol){	
	vpp::pixel_wise(data_.img_, diff) | [&] (const value_type& i, weights_type& w) { w = i.sum(); };
	vpp::pixel_wise(data_.img_, proximal_mappings_[0], proximal_mappings_[1], proximal_mappings_[2], proximal_mappings_[3], proximal_mappings_[4]) | karcher_mean;
	vpp::pixel_wise(data_.img_, diff)(vpp::_no_threads) | [&] (const value_type& i, const weights_type& w) { error = std::max(error, std::abs(w - i.sum())); };
    }
}


template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::prpt_step(double muk){
    weights_mat X = func_.getweightsX();
    weights_mat Y = func_.getweightsY();

    updateFidelity(muk);
    
    updateTV(muk, 0, Y);
    updateTV(muk, 1, X);

    geod_mean();
}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<PRPT, FUNCTIONAL, MANIFOLD, DATA, PAR>::minimize(){
    std::cout << "Starting Proximal Point Algorithm with..." << std::endl;
    std::cout << "\t Lambda = \t" << func_.getlambda() << std::endl;
    std::cout << "\t Tolerance = \t" <<  tolerance_ << std::endl;
    std::cout << "\t Max Steps = \t" << max_prpt_steps_ << std::endl;
    
    
    prpt_step_ = 0;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> t = std::chrono::duration<double>::zero();
    start = std::chrono::system_clock::now();
    
    // PRPT Iteration Loop
    while(prpt_step_ < max_prpt_steps_ && t.count() < max_runtime_){
	
	std::cout << "PRPT Step #" << prpt_step_+1 << std::endl;
	typename FUNCTIONAL::result_type J = func_.evaluateJ();
	Js_.push_back(J);
	std::cout << "\t Value of Functional J: " << J << std::endl;
	
	double muk = 3.0 * std::pow(static_cast<double>(prpt_step_),-0.95);
	
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
