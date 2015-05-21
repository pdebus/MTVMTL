#ifndef TVMTL_TVMINIMIZER_HPP
#define TVMTL_TVMINIMIZER_HPP

//System includes
#include <iostream>
#include <chrono>

//Eigen includes
#include <Eigen/Sparse>

//vpp includes
#include <vpp/vpp.hh>


namespace tvmtl {

// Primary Template
template < enum ALGORITHM AL, class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
    class TV_Minimizer {
    
    };


template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
    class TV_Minimizer< IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR > {
    
	public:
	    // Manifold typedefs
	    typedef typename MANIFOLD::value_type value_type;

	    // Functional typedefs
	    typedef typename FUNCTIONAL::gradient_type gradient_type;
	    typedef typename FUNCTIONAL::sparse_hessian_type hessian_type;
	    typedef typename Eigen::SparseSelfAdjointView<hessian_type, Eigen::Upper> sa_hessian_type;


	    // Parameters from traits class
	    typedef algo_traits< IRLS > AT;
	    static const int runtime = AT::max_runtime;
    
	    typedef double newton_error_type;


	    // Constructor
	    TV_Minimizer(FUNCTIONAL& func, DATA& dat):
		func_(func),
		data_(dat)
	    {
		sparse_pattern_analyzed_ = false;
	    }

	    void first_guess();
	    void smoothening(int smooth_steps);
	    
	    newton_error_type newton_step();
	    void minimize();
	    void output() { std::cout << "OUTPUT TEST" << std::endl; }
	

	private:
	    FUNCTIONAL& func_;
	    DATA& data_;
	 
	    //AT::solver< sa_hessian_type > solver_;
	    AT::solver< hessian_type > solver_;
	    bool sparse_pattern_analyzed_;

	    int irls_step_;
	    int newton_step_;

	    std::vector< std::chrono::duration<double> > Ts_;
	    std::vector< typename FUNCTIONAL::result_type > Js_;

    };

/*----- IMPLEMENTATION------*/


//Smoothening
template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR>::smoothening(int smooth_steps){
    
    std::cout << "Start Smoothening with max_steps = " << smooth_steps << std::endl;

    //TODO: Add Manifold dependent tranformation Before and after smoothening
    typename FUNCTIONAL::result_type Jnew = func_.evaluateJ();
    typename FUNCTIONAL::result_type Jold = Jnew + 1;
    int step = 0;

    typename FUNCTIONAL::img_type temp_img(data_.img_.domain(), vpp::_border=1);
    vpp::copy(data_.img_, temp_img);
    typename FUNCTIONAL::nbh_type N(temp_img);

    while(Jnew<Jold && step < smooth_steps){
	std::cout << "\tSmoothening step #" << step+1 << std::endl;
	std::cout << "\t Value of Functional J: " << Jnew << std::endl;
	Jold = Jnew;
	// Standard smoothening stencil 
	//	1	 
	//  1	4   1	/   8
	//	1      
	auto boxfilter = [&] (value_type& i, const auto& nbh) { 
	    i = (4 * nbh(0,0) + nbh(1,0) + nbh(0,1) + nbh(-1,0) + nbh(0,-1))/8.0; 
	};
	vpp::pixel_wise(data_.img_, N)(/*vpp::_no_threads*/) | boxfilter;
	vpp::copy(data_.img_, temp_img);
	Jnew = func_.evaluateJ();
	step++;
    }

    std::cout << "Smoothening completed.\n\n" << std::endl;

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
typename TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR>::newton_error_type TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR>::newton_step(){

    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    int value_dim = FUNCTIONAL::value_dim;

    // Calculate the gradient and hessian 
    func_.evaluateDJ();
    func_.evaluateHJ();
    
    // Set up the sparse Linear system
    gradient_type x;
    const gradient_type& b = func_.getDJ();
    const hessian_type& A = func_.getHJ();//.template selfadjointView<Eigen::Upper>();

    if (!sparse_pattern_analyzed_){
	solver_.analyzePattern(A);
	sparse_pattern_analyzed_ =  true;	
    }
    
    //TODO: Add optinal Preconditioning here
    // if(AT::use_preconditioner){
    //
    // }

    // Solve the System
    solver_.factorize(A);
    x = solver_.solve(b);
    
    //TODO: Apply tangent space backtransformation here
    

    // Apply Newton correction to picture
    // TODO: 
    // - Generalize to all Manifold: Add functional call to MANIFOLD::exp(p,DJ_.segment(....)) instead of "-="
    // - This is also dimension-dependent 2D or 3D so try moving to functional class
    vpp::pixel_wise(data_.img_, data_.img_.domain()) | [&] (value_type& p, vpp::vint2 coord) { p -= x.segment(3*(coord[0]+nr*coord[1]), value_dim); };

    // Compute the Error
    newton_error_type error = x.norm();

    return error;
}


template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR>::minimize(){
    std::cout << "Starting IRLS Algorithm with..." << std::endl;
    std::cout << "\t Lambda = \t" << func_.getlambda() << std::endl;
    std::cout << "\t eps^2 = \t" << func_.geteps2() << std::endl;
    std::cout << "\t Tolerance = \t" <<  AT::tolerance << std::endl;
    std::cout << "\t Max Steps IRLS= \t" << AT::max_irls_steps << std::endl;
    std::cout << "\t Max Steps Newton = \t" << AT::max_newtons_steps << std::endl;
    
    
    irls_step_ = 0;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> t = std::chrono::duration<double>::zero();
    start = std::chrono::system_clock::now();
    
    // IRLS Iteration Loop
    while(irls_step_ < AT::max_irls_steps && t.count() < AT::max_runtime){
	
	std::cout << "IRLS Step #" << irls_step_+1 << std::endl;
	// NOTE: evaluation of J automatically calls updateWeights(): separate eventually
	typename FUNCTIONAL::result_type J = func_.evaluateJ();
	Js_.push_back(J);
	std::cout << "\t Value of Functional J: " << J << std::endl;
	
	newton_step_ = 0;
	newton_error_type error = AT::tolerance + 1;

	// Newton Iteration Loop
	while(AT::tolerance < error && t.count() < AT::max_runtime && newton_step_ < AT::max_newtons_steps){
	    std::cout << "\t Newton step #" << newton_step_+1;
	    error = newton_step();
	    std::cout << "\t Error: " << error << std::endl;
	    newton_step_++;
	}
	
	end = std::chrono::system_clock::now();
	t = end - start; 
	irls_step_++;
	Ts_.push_back(t);
    }

    std::cout << "Minimization in " << t.count() << " seconds." << std::endl;
}



} // end namespace tvmtl


#endif
