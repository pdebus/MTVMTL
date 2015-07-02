#ifndef TVMTL_TVMINIMIZER_HPP
#define TVMTL_TVMINIMIZER_HPP

//System includes
#include <iostream>
#include <map>
#include <vector>
#include <chrono>

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

// Primary Template
template < enum ALGORITHM AL, class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
    class TV_Minimizer {
    
    };


template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
    class TV_Minimizer< IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR > {
    
	public:
	    // Manifold typedefs
	    typedef typename MANIFOLD::scalar_type scalar_type;
	    typedef typename MANIFOLD::value_type value_type;
	    typedef typename MANIFOLD::tm_base_type tm_base_type;

	    // Functional typedefs
	    typedef typename FUNCTIONAL::gradient_type gradient_type;
	    typedef typename FUNCTIONAL::sparse_hessian_type hessian_type;
	    typedef typename FUNCTIONAL::tm_base_mat_type tm_base_mat_type;
	    typedef typename Eigen::SparseSelfAdjointView<hessian_type, Eigen::Upper> sa_hessian_type;


	    // Parameters from traits class
	    typedef algo_traits< MANIFOLD::MyType > AT;
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
	 
	    typename AT::template solver< hessian_type > solver_;
	    bool sparse_pattern_analyzed_;

	    int irls_step_;
	    int newton_step_;

	    std::vector< std::chrono::duration<double> > Ts_;
	    std::vector< typename FUNCTIONAL::result_type > Js_;

    };

/*----- IMPLEMENTATION------*/

//First Guess
template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR>::first_guess(){

    std::cout << "Starting interpolation of damaged Area" << std::endl;

    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef CGAL::Delaunay_triangulation_2<K> Delaunay_triangulation;
    typedef CGAL::Interpolation_traits_2<K> Traits;
    typedef K::FT Coord_type;
    typedef K::Point_2 Point;

    typedef typename DATA::inp_type inp_type;
    int value_dim = FUNCTIONAL::value_dim;
    int value_rows = value_type::RowsAtCompileTime;
    int value_cols = value_type::ColsAtCompileTime;
    
    for(int r=0; r<value_rows; r++)
	for(int c=0; c<value_cols; c++){
	    std::cout << "\t Channel " << value_cols*r+c+1 << " of " << value_dim << "..." << std::endl;
	    Delaunay_triangulation T;
	    std::map<Point, Coord_type, K::Less_xy_2> function_values;
	    typedef CGAL::Data_access< std::map<Point, Coord_type, K::Less_xy_2 > >  Value_access;
	    #ifdef TVMTL_TVMIN_DEBUG
		std::cout << "NZ-Entries in inpainting matrix:" << vpp::sum(data_.inp_) << std::endl;
	    #endif

	    int numnodes=0;
	    // Add Interpolation nodes
	    vpp::pixel_wise(data_.inp_, data_.img_, data_.img_.domain())(vpp::_no_threads)  | [&] (inp_type inp, const value_type& i, const vpp::vint2& coord) {
		if(!inp){
			Point p(coord[0], coord[1]);
			T.insert(p);
			function_values.insert(std::make_pair(p,i(r,c)));
			numnodes++;
		}
	    };
	    std::cout << "\t\tNumber of Nodes: " << numnodes << std::endl;

	    int numdampix=0;
	    // Interpolate missing nodes
	    vpp::pixel_wise(data_.inp_, data_.img_, data_.img_.domain())(vpp::_no_threads) | [&] (inp_type inp, value_type& i, const vpp::vint2& coord) {
		if(inp){
		    Point p(coord[0], coord[1]);
		    std::vector< std::pair< Point, Coord_type > > coords;
                    Coord_type norm =  CGAL::natural_neighbor_coordinates_2(T, p,std::back_inserter(coords)).second;
		    Coord_type res = CGAL::linear_interpolation(coords.begin(), coords.end(), norm,Value_access(function_values));
		    i(r,c)=static_cast<scalar_type>(res);
		    numdampix++;
		}
	    };
	    std::cout << "\t\tNumber of interpolated Pixels: " << numdampix << std::endl;
	}

	vpp::pixel_wise(data_.img_) | [&] (value_type& i) { MANIFOLD::projector(i); };

}

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
	    MANIFOLD::projector(i);
	};

	vpp::pixel_wise(data_.img_, N)(/*vpp::_no_threads*/) | boxfilter;
	//data_.output_matval_img("son_img2.csv");
	vpp::copy(data_.img_, temp_img);
	Jnew = func_.evaluateJ();
	step++;
    }

    std::cout << "Smoothening completed with J=" << Jnew << "\n\n" << std::endl;

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
typename TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR>::newton_error_type TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR>::newton_step(){

    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
    int value_dim = FUNCTIONAL::value_dim;
    int manifold_dim = FUNCTIONAL::manifold_dim;

    // Calculate the gradient and hessian 
    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
	std::cout << "\n\t\t...Calculate Gradient" << std::endl;
    #endif    
    func_.evaluateDJ();
   

    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
	std::cout << "\t\t...Calculate Hessian" << std::endl;
    #endif
    func_.evaluateHJ();

 

    // Set up the sparse Linear system
    gradient_type x;
    const gradient_type& b = func_.getDJ();
    const hessian_type& A = func_.getHJ();//.template selfadjointView<Eigen::Upper>();

    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
	std::cout << "\t\t\t...Gradient Size: " << b.size() << std::endl; 
	std::cout << "\t\t\t...Hessian Non-Zeros: " << A.nonZeros() << std::endl; 
	std::cout << "\t\t\t...Hessian Rows: " << A.rows() << std::endl; 
	std::cout << "\t\t\t...Hessian Cols: " << A.cols() << std::endl; 
	std::cout << "\n\t\t...Analyze Sparse Pattern" << std::endl;
    #endif
    if (!sparse_pattern_analyzed_){
	solver_.analyzePattern(A);
	sparse_pattern_analyzed_ =  true;	
    }
    
    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
	std::cout << "\t\t...Solve System" << std::endl;
    #endif

    // Solve the System
    solver_.factorize(A);
    x = solver_.solve(b);
    
    // Apply Newton correction to picture
    // TODO: 
    // - This is also dimension-dependent 2D or 3D so try moving to functional class
    // - Change VectorXd to something parametrized with scalar_type
    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
	std::cout << "\t\t...Apply Newton Correction" << std::endl;
    #endif
    const tm_base_mat_type& T = func_.getT();
    auto newton_correction = [&] (const tm_base_type& t, value_type& i, const vpp::vint2 coord) { 
	Eigen::VectorXd v = -t*x.segment(manifold_dim*(coord[0]+nr*coord[1]), manifold_dim);
	MANIFOLD::exp(i, Eigen::Map<value_type>(v.data()), i);
	//MANIFOLD::exp(i, -t*x.segment(manifold_dim*(coord[0]+nr*coord[1]), manifold_dim), i);
    };
    vpp::pixel_wise(T, data_.img_, data_.img_.domain()) | newton_correction;
    // Compute the Error
     #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
	std::cout << "\t\t...Compute Newton error" << std::endl;
    #endif
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
	std::cout << "\t Elapsed time: " << t.count() << " seconds." << std::endl;
	irls_step_++;
	Ts_.push_back(t);
    }

    std::cout << "Minimization in " << t.count() << " seconds." << std::endl;
}



} // end namespace tvmtl


#endif
