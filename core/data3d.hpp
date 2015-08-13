#ifndef TVTML_DATA3D_HPP
#define TVTML_DATA3D_HPP

// system includes
#include <cassert>
#include <limits>
#include <cmath>
#include <random>

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

//Eigen includes
#include <Eigen/Core>
#include <Eigen/Geometry>


// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef TV_DATA_DEBUG
    #include <opencv2/highgui/highgui.hpp>
#endif

// video++ includes
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

// own includes 
#include "data3d_utils.hpp"
#include "manifold.hpp"

namespace tvmtl{

// Specialization 3D Data
template < typename MANIFOLD >
class Data<MANIFOLD, 3>{
    
    public:
	static const int img_dim;

	// Manifold typedefs
	typedef typename MANIFOLD::value_type value_type;
	typedef typename MANIFOLD::scalar_type scalar_type;
	
	// Storage typedefs
	typedef vpp::image3d<value_type> storage_type;
	
	typedef double weights_type;
	typedef vpp::image3d<weights_type> weights_mat;
	
	typedef bool inp_type;
	typedef vpp::image3d<inp_type> inp_mat;

	inline bool doInpaint() const { return inpaint_; }
	    
	// Data Init functions
	inline void initEdgeweights();
	inline void initInp();
	
	// Input functions
	void rgb_slice_reader(const char* filename, int num_slides);
	void readMatrixDataFromCSV(const char* filename, const int nz, const int ny, const int nx);
	
	// Creation functions
	void create_noisy_gray(const int nz, const int ny, const int nx, double color=0.5, double stdev=0.1);
	void create_noisy_rgb(const int nz, const int ny, const int nx, int color=1, double stdev=0.1);

	void setEdgeWeights(const weights_mat&);

//  private:
	storage_type img_;
	storage_type noise_img_;
	weights_mat edge_weights_;

	bool inpaint_;
	inp_mat inp_; 
};


/*----- Implementation 3D Data ------*/
template < typename MANIFOLD >
const int Data<MANIFOLD, 3>::img_dim = 3;

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::setEdgeWeights(const weights_mat& w){
    clone3d(w, edge_weights_);
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::initInp(){
    inp_ = inp_mat(noise_img_.domain());
    fill3d(inp_, false);
    inpaint_ = false;
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::initEdgeweights(){
    edge_weights_ = weights_mat(noise_img_.domain());
    fill3d(edge_weights_, 1.0);
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::create_noisy_gray(const int nz, const int ny, const int nx, double color, double stdev){
    static_assert(MANIFOLD::value_dim == 1, "Method is only callable for Manifolds with embedding dimension 1");
    noise_img_ = storage_type(nz, ny, nx);
    img_ = storage_type(noise_img_.domain());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<typename MANIFOLD::scalar_type> rand(0.0, stdev);
    auto insert = [&] (value_type& i) { i.setConstant(color + rand(gen)); };

    pixel_wise3d(insert, noise_img_);
    clone3d(noise_img_, img_);
    initInp();
    initEdgeweights();
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::create_noisy_rgb(const int nz, const int ny, const int nx, int color, double stdev){
    static_assert(MANIFOLD::value_dim == 3, "Method is only callable for Manifolds with embedding dimension 3");
    noise_img_ = storage_type(nz, ny, nx);
    img_ = storage_type(noise_img_.domain());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<typename MANIFOLD::scalar_type> rand(0.0, stdev);
    value_type v(0.7 + rand(gen), 0.3 + rand(gen), 0.3 + rand(gen));
    auto insert = [&] (value_type& i) { i = v;  };

    pixel_wise3d(insert, noise_img_);
    clone3d(noise_img_, img_);
    initInp();
    initEdgeweights();
}

template < typename MANIFOLD >
void Data<MANIFOLD, 3>::rgb_slice_reader(const char* filename, int num_slices){
	static_assert(MANIFOLD::value_dim == 3,"ERROR: RGB Input requires a Manifold with embedding dimension N=3!");
	std::string fname(filename);

	int last_dot_pos = fname.find_last_of(".");
	std::string basefilename  = fname.substr(0, last_dot_pos-1);
	std::string ext = fname.substr(last_dot_pos, fname.length());
	

	vpp::image2d<vpp::vuchar3> input_image = vpp::clone(vpp::from_opencv<vpp::vuchar3 >(cv::imread(basefilename + std::to_string(0) + ext)));
	vpp::image2d<value_type> input_image_double(input_image.domain());
	int nr = input_image.nrows();
	int nc = input_image.ncols();
	noise_img_ = storage_type(num_slices, nr, nc);

	#ifdef TV_DATA_DEBUG
	    std::cout << "\nReading slice sequence " << basefilename + std::to_string(0) + ext << " containing " << num_slices << " Slices. " << std::endl;
	    std::cout << "Dimensions (Slices, Rows, Cols): " << num_slices << " X " << nr << " X " << nc << std::endl;
	#endif
		  
	// Convert Picture of uchar to double 
	auto converter =  [] (const auto& i, auto& d) {
	    value_type v = value_type::Zero();
	    vpp::vuchar3 vu = i;
	    // TODO: insert manifold scalar type
	    v[0]=static_cast<double>(vu[2]); //opencv saves as BGR
	    v[1]=static_cast<double>(vu[1]);
	    v[2]=static_cast<double>(vu[0]);
	    d = v / static_cast<double>(std::numeric_limits<unsigned char>::max());
	};
	
	for(int s = 0; s < num_slices; ++s ){
	#ifdef TV_DATA_DEBUG
	    std::cout << "Reading slice number " << s << " with name " << basefilename + std::to_string(s) + ext << std::endl;
	#endif
	    input_image = vpp::clone(vpp::from_opencv<vpp::vuchar3 >(cv::imread(basefilename + std::to_string(s) + ext)));
	    vpp::pixel_wise(input_image, input_image_double) | converter; 
	    
	    #pragma omp parallel for
	    for(int r = 0; r < nr; ++r){
		value_type* row_pointer3d = &noise_img_(s, r, 0);
		value_type* row_pointer2d = &input_image_double(r,0);
		for(int c = 0; c < nc; ++c)
		    row_pointer3d[c] = row_pointer2d[c];
	    }
	}
    img_ = storage_type(noise_img_.domain());
    clone3d(noise_img_, img_);

    initInp();
    initEdgeweights();
}


template <typename MANIFOLD>
void Data<MANIFOLD, 3>::readMatrixDataFromCSV(const char* filename, const int nz, const int ny, const int nx){
    #ifdef TV_DATA_DEBUG
	std::cout << "ReadMatrixData from CSV File..." << std::endl;
    #endif
    noise_img_ = storage_type(nz, ny, nx);
    fill3d(noise_img_, MANIFOLD::value_type::Zero());

    const int N = MANIFOLD::value_type::RowsAtCompileTime; 
    const int N2 = MANIFOLD::value_dim;
    int cols, rows = 0;

    std::ifstream infile(filename, std::ifstream::in);
    
    std::string line = "";
    while (std::getline(infile, line)){
	std::stringstream strstr(line);
	std::string word = "";
	
	if(cols == 0)
	    while (std::getline(strstr, word, ','))
		cols++;
	
	rows++;
    }

    assert(N2 == cols);
    assert(nz*nx*ny == rows);
    
    infile.clear();
    infile.seekg(0, std::ios_base::beg);
    
    Eigen::Matrix<typename MANIFOLD::scalar_type, N2, 1> vectorizedMat;
    vectorizedMat.setZero();

    auto it = noise_img_.begin();
    while (std::getline(infile, line)){
	std::stringstream strstr(line);
	std::string word = "";
	int j=0;
	while (std::getline(strstr, word,',')){
	    typename MANIFOLD::scalar_type entry= static_cast<typename MANIFOLD::scalar_type>(std::stod(word));
	    vectorizedMat(j) = entry;
	    ++j;
	}
	*it = Eigen::Map<typename MANIFOLD::value_type>(vectorizedMat.data());
	it.next();
    }
    img_ = storage_type(noise_img_.domain()); 
    clone3d(noise_img_, img_);

    initInp();
    initEdgeweights();
}



}// end namespace tvmtl

#endif
