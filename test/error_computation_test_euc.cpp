#include <iostream>
#include <string>
#include <sstream>

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"

int main(int argc, const char *argv[])
{
	using namespace tvmtl;
	
	typedef Manifold< EUCLIDIAN, 3 > mf_t;
	typedef Data< mf_t, 2> data_t;	
	typedef Functional<FIRSTORDER, ANISO, mf_t, data_t> func_t;
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;

	data_t solution = data_t();
	data_t myData = data_t();

	myData.rgb_imread(argv[1]);
	myData.add_gaussian_noise(0.2);
	int ny = myData.img_.nrows();
	int nx = myData.img_.ncols();
	
	typename data_t::storage_type copy(myData.img_.domain());
	copy = vpp::clone(myData.img_);

	std::string fname(argv[1]);
	std::string statfile_name, solfile_name;
	
	std::stringstream sstream;

	sstream << fname << "_" << ny << "x" << nx;
	statfile_name = sstream.str() + "_IRLSstats.csv";
	solfile_name = sstream.str() + "_sol.csv";
	if(argc != 3){
		double lam=0.2;
		func_t myFunc(lam, myData);
		myFunc.seteps2(1e-16);

		tvmin_t myTVMin(myFunc, myData);
		myTVMin.setMax_irls_steps(20);
		myTVMin.setMax_runtime(10000);
		myTVMin.minimize();
		myData.output_matval_img(solfile_name.c_str());
		std::cout << "Minimizer data saved to " << solfile_name << std::endl;
		return 0;
	}

	solution.readMatrixDataFromCSV(argv[2], nx, ny);

	double lam=0.2;
	func_t myFunc(lam, myData);
	
	
	// ====================================IRLS======================================================
	myFunc.seteps2(1e-16);
	tvmin_t myTVMin(myFunc, myData);

	//std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
	//myTVMin.smoothening(5);
 
	int irls_step_ = 0;
	int max_runtime_ = 10000;
	int max_irls_steps_ = 15;
	int max_newton_steps_ = 1;
	double tolerance_ = 1e-4;

	std::cout << "Starting IRLS Algorithm with..." << std::endl;
	std::cout << "\t Lambda = \t" << myFunc.getlambda() << std::endl;
	std::cout << "\t eps^2 = \t" << myFunc.geteps2() << std::endl;
	std::cout << "\t Tolerance = \t" <<  tolerance_ << std::endl;
	std::cout << "\t Max Steps IRLS= \t" << max_irls_steps_ << std::endl;
	std::cout << "\t Max Steps Newton = \t" << max_newton_steps_ << std::endl;
    
   	std::fstream f;
	f.open(statfile_name,std::fstream::out);

	typename func_t::result_type J = myFunc.evaluateJ();
	double totalerror = vpp::sum( vpp::pixel_wise(myData.img_, solution.img_) | [](const auto& i, const auto& s) {return mf_t::dist_squared(i,s);} );
	
	std::cout << irls_step_ << ", " << 0 << ", " << J << ", " << totalerror << ", " << 0 << std::endl;
	f << irls_step_ << ", " << 0 << ", " << J << ", " << totalerror << ", " << 0 << std::endl;

        std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> t = std::chrono::duration<double>::zero();
	start = std::chrono::system_clock::now();
    
	// IRLS Iteration Loop
	while(irls_step_ < max_irls_steps_){
	    
	    int newton_step_ = 0;
	    typename tvmin_t::newton_error_type error = tolerance_ + 1;

	    // Newton Iteration Loop
	    while(tolerance_ < error && t.count() < max_runtime_ && newton_step_ < max_newton_steps_){
		error = myTVMin.newton_step();
		newton_step_++;
	    }

	    J = myFunc.evaluateJ();
	    totalerror = vpp::sum( vpp::pixel_wise(myData.img_, solution.img_) | [](const auto& i, const auto& s) {return mf_t::dist_squared(i,s);} );

	    end = std::chrono::system_clock::now();
	    t = end - start; 
	    double seconds = t.count();
	    std::cout << irls_step_+1 << ", " << newton_step_ << ", " << J << ", " << totalerror << ", " << seconds << std::endl;
	    f << irls_step_+1 << ", " << newton_step_ << ", " << J << ", " << totalerror << ", " << seconds << std::endl;
	    irls_step_++;
	}
	f.close();
	std::cout << "IRLS Minimization in " << t.count() << " seconds." << std::endl;

	//===================================== PROXIMAL POINT====================================================
	myData.img_ = vpp::clone(copy);

	typedef TV_Minimizer< PRPT, func_t, mf_t, data_t, OMP > prpt_t;
	myFunc.seteps2(0.0);
	prpt_t myPRPT(myFunc, myData);

	int prpt_step_ = 1;
	int max_prpt_steps_ = 500;
	max_runtime_ = 10000;

	statfile_name = sstream.str() + "_PRPTstats.csv";
	std::cout << "Starting Proximal Point Algorithm with..." << std::endl;
	std::cout << "\t Lambda = \t" << myFunc.getlambda() << std::endl;
	std::cout << "\t Max Steps = \t" << max_prpt_steps_ << std::endl;
    
    	f.open(statfile_name,std::fstream::out);
	
	J = myFunc.evaluateJ();
	totalerror = vpp::sum( vpp::pixel_wise(myData.img_, solution.img_) | [](const auto& i, const auto& s) {return mf_t::dist_squared(i,s);} );
	std::cout << 0 << ", " << 0 << ", " << J << ", " << totalerror << ", " << 0 << std::endl;
	f << 0 << ", " << 0 << ", " << J << ", " << totalerror << ", " << 0 << std::endl;

	t = std::chrono::duration<double>::zero();
	start = std::chrono::system_clock::now();
    
	// PRPT Iteration Loop
	while(prpt_step_ <= max_prpt_steps_ && t.count() < max_runtime_){
	    
	    double muk = 3.0 * std::pow(static_cast<double>(prpt_step_),-0.95);

	    myPRPT.prpt_step(muk);

	    J = myFunc.evaluateJ();
	    totalerror = vpp::sum( vpp::pixel_wise(myData.img_, solution.img_) | [](const auto& i, const auto& s) {return mf_t::dist_squared(i,s);} );
	    
	    end = std::chrono::system_clock::now();
	    t = end - start; 
	    double seconds = t.count();
	    std::cout << prpt_step_ << ", " << 0 << ", " << J << ", " << totalerror << ", " << seconds << std::endl;
	    f << prpt_step_ << ", " << 0 << ", " << J << ", " << totalerror << ", " << seconds << std::endl;
	    prpt_step_++;
	}

	std::cout << " PRPT Minimization in " << t.count() << " seconds." << std::endl;

	return 0;
}
