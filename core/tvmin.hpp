#ifndef TVMTL_TVMINIMIZER_HPP
#define TVMTL_TVMINIMIZER_HPP

#include <iostream>

#include <Eigen/Sparse>


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
	    typedef typename FUNCTIONAL::sparse_hessian_type hessian_type;
	    typedef typename FUNCTIONAL::gradient_type gradient_type;


	    // Parameters from traits class
	    typedef algo_traits< IRLS > AT;
	    static const int runtime = AT::max_runtime;
    
	    typedef double newton_error_type;


	    // Constructor
	    TV_Minimizer(FUNCTIONAL func, DATA dat):
		func_(func),
		data_(dat)
	    {}

	    void first_guess();
	    void smoothening();
	    
	    newton_error_type newton_step();
	    void minimize();
	    void output();

	private:
	    FUNCTIONAL func_;
	    DATA& data_;
	 
	    AT::solver< hessian_type > solver_;
	    bool sparse_pattern_analyzed_;

	    int irls_step_;
	    int newton_step_;
    };

/*----- IMPLEMENTATION------*/


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
    const gradient_type& b = func_.DJ_;
    const hessian_type& A = func_.HJ_.selfadjointView<Eigen::Upper>();

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
    // TODO: Generalize to all Manifold: Add functional call to MANIFOLD::exp(p,DJ_.segment(....)) instead of "-="
    vpp::pixel_wise(data_.img_, data_.img_.domain()) | [&] (value_type& p, vpp::vint2 coord) { p -= x.segment(3*(coord[0]+nr*coord[1]), value_dim); };

    // Compute the Error
    newton_error_type error = x.norm();
    
    newton_step_++;

    return error;
}


template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR>::minimize(){
    irls_step_ = 0;

    time_diff t=0;
    time tstart=now();
    
    // IRLS Iteration Loop
    while(irls_step_ < AT::max_irls_steps && t < AT::max_runtime){
	// NOTE: evaluation of J automatically calls updateWeights(): separate eventually
	typename FUNCTIONAL::result_type J = func_.evaluateJ();
	Js.push_back(J);

	newton_step_ = 0;
	newton_error_type error = AT::tolerance + 1;

	// Newton Iteration Loop
	while(AT::tolerance < error && t < AT::max_runtime && newton_step_ < max_newtons_steps)
	    error = newton_step();
	
	t = now()-tstart;
	Ts.push_back(t);
    }


}



} // end namespace tvmtl


#endif
