#ifndef TVMTL_TVMINIMIZER_HPP
#define TVMTL_TVMINIMIZER_HPP

namespace tvmtl {

// Primary Template
template < enum ALGORITHM AL, Functional, Manifold, Data, LA_HANDLER, PAR > 
    class TV_Minimizer {
    
    };


template < Functional, Manifold, Data, LA_HANDLER, PAR > 
    class TV_Minimizer< IRLS > {
    
	public:
	    static const int MyType = IRLS;

	    static const int N = Manifold::dim;
	    static const int N = Manifold::scalar_type;

	    // Linear Algebra typedefs
	    typedef typename LA_HANDLER::MAT<scalar_type, N> mat_type;

	    // Parameters from traits class
	    typedef Algo_traits< MyType > AT;
	    static const int runtime = AT::max_runtime;
    
	    // Constructor
	    TV_Minimizer(functional func, Data dat):
		func_(func),
		data_(dat)
	    {}

	    void minimize();
	    void output();

	private:
	    void first_guess();
	    void smoothening();
	    void newton_step();
	    const& mat_type IRLS_weights(...);

	    Functional func_;
	    data& data_;
    };


template < Functional, Manifold, Data, LA_HANDLER, PAR > 
    class TV_Minimizer< PRPT > {

    };

} // end namespace tvmtl


#endif
