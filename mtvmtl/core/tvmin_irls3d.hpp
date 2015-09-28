#ifndef TVMTL_TVMINIMIZER_IRLS3D_HPP
#define TVMTL_TVMINIMIZER_IRLS3D_HPP

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

//CGAL includes For linear Interpolation
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_3.h>
#include <CGAL/interpolation_functions.h>


//vpp includes
#include <vpp/vpp.hh>


namespace tvmtl {

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
    class TV_Minimizer< IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR, 3> {
    
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
		max_runtime_=AT::max_runtime;
		max_irls_steps_=AT::max_irls_steps;
		max_newton_steps_=AT::max_newton_steps;
		tolerance_=AT::tolerance;
		sparse_pattern_analyzed_ = false;
	    }

	    void first_guess();
	    void smoothening(int smooth_steps);
	    
	    newton_error_type newton_step();
	    void minimize();
	    void output() { std::cout << "OUTPUT TEST" << std::endl; }

	    void setMax_runtime(int t) { max_runtime_ = t; }
	    void setMax_irls_steps(int n) { max_irls_steps_ = n; }
	    void setMax_newton_steps(int n) { max_newton_steps_ = n; }
	    void setTolerance(double t) {tolerance_ =t; }
	    
	    int max_runtime(int t) const { return max_runtime_; }
	    int  max_irls_steps(int n) const { return max_irls_steps_; }
	    int max_newton_steps(int n) const { return max_newton_steps_; }
	    int tolerance(double t) const { return tolerance_; }
	

	private:
	    FUNCTIONAL& func_;
	    DATA& data_;
	 
	    typename AT::template solver< hessian_type > solver_;
	    bool sparse_pattern_analyzed_;

	    int irls_step_;
	    int newton_step_;

	    int max_runtime_;
	    int max_irls_steps_;
	    int max_newton_steps_;
	    double tolerance_;

	    std::vector< std::chrono::duration<double> > Ts_;
	    std::vector< typename FUNCTIONAL::result_type > Js_;

    };

/*----- IMPLEMENTATION IRLS------*/

//First Guess
template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::first_guess(){
    
    int ns = data_.img_.nslices();
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();

    std::cout << "Starting interpolation of damaged Area" << std::endl;

    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef CGAL::Delaunay_triangulation_3<K> Delaunay_triangulation;
    typedef CGAL::Interpolation_traits_2<K> Traits;
    typedef K::FT Coord_type;
    typedef K::Point_3 Point;

    typedef typename DATA::inp_type inp_type;
    int value_dim = FUNCTIONAL::value_dim;
    int value_rows = value_type::RowsAtCompileTime;
    int value_cols = value_type::ColsAtCompileTime;

    /*
    if(MANIFOLD::non_isometric_embedding)
        pixel_wise3d([&] (value_type& i){ MANIFOLD::interpolation_preprocessing(i); }, data_.img_);
     
    for(int R=0; R<value_rows; R++)
	for(int C=0; C<value_cols; C++){
	    std::cout << "\t Channel " << value_cols*R+C+1 << " of " << value_dim << "..." << std::endl;
	    Delaunay_triangulation T;
	    std::map<Point, Coord_type, K::Less_xy_2> function_values;
	    typedef CGAL::Data_access< std::map<Point, Coord_type, K::Less_xy_2 > >  Value_access;
	    #ifdef TVMTL_TVMIN_DEBUG
		std::cout << "NZ-Entries in inpainting matrix:" << sum3d(data_.inp_) << std::endl;
	    #endif

	    int numnodes=0;
		
	    // Add Interpolation nodes
	    for(int s = 0; s < ns; ++s){
		for(int r = 0; r < nr; ++r){
		// Start of row pointers
		const inp_type* inp = &data_.inp_(s, r, 0);
		const value_type* i = &data_.img_(s, r, 0);
		for(int c = 0; c < nc; ++c)
		    if(!inp[c]){
			Point p(s, r, c);
			T.insert(p);
			function_values.insert(std::make_pair(p,i[c](R,C)));
			numnodes++;
		    }	    
		}
	    }

	    std::cout << "\t\tNumber of Nodes: " << numnodes << std::endl;

	    int numdampix=0;
	    // Interpolate missing nodes
	    for(int s = 0; s < ns; ++s){
		for(int r = 0; r < nr; ++r){
		    // Start of row pointers
		    const inp_type* inp = &data_.inp_(s, r, 0);
		    const value_type* i = &data_.img_(s, r, 0);
		    for(int c = 0; c < nc; ++c)
			if(inp[c]){
			    Point p(s, r, c);
		            std::vector< std::pair< Point, Coord_type > > coords;
			    Coord_type norm =  CGAL::natural_neighbor_coordinates_2(T, p,std::back_inserter(coords)).second;
			    Coord_type res = CGAL::linear_interpolation(coords.begin(), coords.end(), norm,Value_access(function_values));
			    i[c](R,C)=static_cast<scalar_type>(res);
			    numdampix++;
			}
		}
	    }

	    std::cout << "\t\tNumber of interpolated Pixels: " << numdampix << std::endl;
	}

	if(MANIFOLD::non_isometric_embedding)
	    pixel_wise3d([&] (value_type& i){ MANIFOLD::interpolation_postprocessing(i); }, data_.img_);

	pixel_wise3d([&] (value_type& i){ MANIFOLD::projector(i); }, data_.img_);
*/
}

//Smoothening
template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::smoothening(int smooth_steps){
    int ns = data_.img_.nslices();
    int nr = data_.img_.nrows();
    int nc = data_.img_.ncols();
 
    std::cout << "Start Smoothening with max_steps = " << smooth_steps << std::endl;

    //TODO: Add Manifold dependent tranformation Before and after smoothening
    typename FUNCTIONAL::result_type Jnew = func_.evaluateJ();
    typename FUNCTIONAL::result_type Jold = Jnew + 1;
    int step = 0;

    typename FUNCTIONAL::img_type temp_img(data_.img_.domain(), vpp::_border=1);
    //    typename FUNCTIONAL::nbh_type N(temp_img);

    std::cout << "Initial functional value J=" << Jnew << "\n\n" << std::endl;

    while(Jnew<Jold && step < smooth_steps){
	if(MANIFOLD::non_isometric_embedding)
	    pixel_wise3d([&] (value_type& i){ MANIFOLD::interpolation_preprocessing(i); }, data_.img_);
	
	clone3d(data_.img_, temp_img);
	//vpp::fill_border_closest(temp_img);

	std::cout << "\tSmoothen step #" << step+1 << std::endl;
	std::cout << "\t Value of Functional J: " << Jnew << std::endl;
	Jold = Jnew;

	const value_type* t = &temp_img(0,0,0);
        for(int s = 0; s < ns; ++s){
	    for(int r = 0; r < nr; ++r){
	     // Start of row pointers
	        value_type* i = &data_.img_(s, r, 0);
	        for(int c = 0; c < nc; ++c){
		    i[c] = (6 * t[s,r,c] + t[s,r,c-1] + t[s,r,c+1] + t[s,r-1,c] + t[s,r+1,c] + t[s-1,r,c] + t[s+1,r,c]) / 12.0;
		}
	    }
	}
	
	if(MANIFOLD::non_isometric_embedding)
	    pixel_wise3d([&] (value_type& i){ MANIFOLD::interpolation_postprocessing(i); }, data_.img_);
	
	Jnew = func_.evaluateJ();
	step++;
    }
    
    std::cout << "Smoothening completed with J=" << Jnew << "\n\n" << std::endl;

}

template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
typename TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::newton_error_type TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::newton_step(){

    int ns = data_.img_.nslices();
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
    // - Change VectorXd to something parametrized with scalar_type
    #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
	std::cout << "\t\t...Apply Newton Correction" << std::endl;
    #endif
    const tm_base_mat_type& T = func_.getT();

    for(int s = 0; s < ns; ++s){
	#pragma omp parallel for
	for(int r = 0; r < nr; ++r){
	    // Start of row pointers
	    value_type* i = &data_.img_(s, r, 0);
	    const tm_base_type* t = &T(s, r, 0);
	    for(int c = 0; c < nc; ++c){
		Eigen::VectorXd v = -t[c]*x.segment(manifold_dim * (c + nc * r + nr * nc * s), manifold_dim); // row_wise coordinates
		MANIFOLD::exp(i[c], Eigen::Map<value_type>(v.data()), i[c]);
	    }
	}
    } 
       // Compute the Error
     #ifdef TVMTL_TVMIN_DEBUG_VERBOSE
	std::cout << "\t\t...Compute Newton error" << std::endl;
    #endif
    newton_error_type error = x.norm();

    return error;
}


template <class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR> 
void TV_Minimizer<IRLS, FUNCTIONAL, MANIFOLD, DATA, PAR, 3>::minimize(){
    std::cout << "Starting IRLS Algorithm with..." << std::endl;
    std::cout << "\t Lambda = \t" << func_.getlambda() << std::endl;
    std::cout << "\t eps^2 = \t" << func_.geteps2() << std::endl;
    std::cout << "\t Tolerance = \t" <<  tolerance_ << std::endl;
    std::cout << "\t Max Steps IRLS= \t" << max_irls_steps_ << std::endl;
    std::cout << "\t Max Steps Newton = \t" << max_newton_steps_ << std::endl;
    
    
    irls_step_ = 0;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> t = std::chrono::duration<double>::zero();
    start = std::chrono::system_clock::now();
    
    // IRLS Iteration Loop
    while(irls_step_ < max_irls_steps_ && t.count() < max_runtime_){
	
	std::cout << "IRLS Step #" << irls_step_+1 << std::endl;
	// NOTE: evaluation of J automatically calls updateWeights(): separate eventually
	typename FUNCTIONAL::result_type J = func_.evaluateJ();
	Js_.push_back(J);
	std::cout << "\t Value of Functional J: " << J << std::endl;
	
	newton_step_ = 0;
	newton_error_type error = tolerance_ + 1;

	// Newton Iteration Loop
	while(tolerance_ < error && t.count() < max_runtime_ && newton_step_ < max_newton_steps_){
	    std::cout << "\t Newton step #" << newton_step_+1;
	    error = newton_step();
	    #ifdef TVMTL_TVMIN_DEBUG
		    std::string fname("step_img.csv");
		    fname = std::to_string(irls_step_) + "." + std::to_string(newton_step_) + fname;
	    #endif

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
