#ifndef TVTML_DATA_HPP
#define TVTML_DATA_HPP

// system includes
#include <cassert>
#include <limits>
#include <cmath>
#include <random>

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef TV_DATA_DEBUG
    #include <opencv2/highgui/highgui.hpp>
#endif

//Eigen includes
#include <Eigen/Core>
#include <Eigen/Geometry>

// video++ includes
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

// own includes 
#include "manifold.hpp"

namespace tvmtl{

// Primary Template
template <typename MANIFOLD, int DIM >
class Data {
};

// Specialization 2D Data
template < typename MANIFOLD >
class Data< MANIFOLD, 2>{

    public:
	static const int img_dim;
	// Manifold typedefs
	typedef typename MANIFOLD::value_type value_type;
	

	// Storage typedefs
	typedef vpp::image2d<value_type> storage_type;
	
	typedef double weights_type;
	typedef vpp::image2d<weights_type> weights_mat;
	
	typedef bool inp_type;
	typedef vpp::image2d<inp_type> inp_mat;

	inline bool doInpaint() const { return inpaint_; }
	    
	// Data Init functions
	inline void initWeights();
	inline void initEdgeWeights();
	inline void initInp();

	// Input functions
	void rgb_imread(const char* filename); 
	void rgb_readBrightness(const char* filename); 
	void rgb_readChromaticity(const char* filename); 
	
	void readMatrixDataFromCSV(const char* filename, const int nx, const int ny);
	
	// Random Input functions
	// TODO: Paramaterize for manifold type
	void create_nonsmooth_son(const int ny, const int nx);

	// EdgeFunctions
	void findEdgeWeights();
	void setEdgeWeights(const weights_mat&);

	// Inpainting Functions
	void findInpWeights(const int channel=2);
	void createRandInpWeights(const double threshold);


	// OutputFunctions
	void output_weights(const weights_mat& mat, const char* filename) const;
	
	void output_img(const char* filename) const;
	void output_matval_img(const char* filename) const;
	void output_nimg(const char* filename) const;
//    private:
	// Data members
	// TODO Don't forget to initialize with 1px border
	// alignment defaults to 16byte for SSE/SSE2, 32 Byte for AVX
	storage_type img_;
	storage_type noise_img_;
	weights_mat weights_;
	weights_mat edge_weights_;

	bool inpaint_;
	inp_mat inp_; 
};


// Specialization 3D Data
template < typename MANIFOLD >
class Data<MANIFOLD, 3>{

};


/*----- Implementation 2D Data ------*/
template < typename MANIFOLD >
const int Data<MANIFOLD, 2>::img_dim = 2;

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::setEdgeWeights(const weights_mat& w){
    edge_weights_= vpp::clone(w);
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::initWeights(){
    weights_ = weights_mat(noise_img_.domain());
    vpp::fill(weights_, 1.0);
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::initInp(){
    inp_ = inp_mat(noise_img_.domain());
    vpp::fill(inp_, false);
    inpaint_ = false;
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::initEdgeWeights(){
    edge_weights_ = weights_mat(noise_img_.domain());
    vpp::fill(edge_weights_, 1.0);
}


// FIXME: This is problematic for greyscale pictures since vuchar3 and vu[i] is hardcoded, works only due to range check of Eigen for []
template < typename MANIFOLD >
void Data<MANIFOLD, 2>::findEdgeWeights(){
    #ifdef TV_DATA_DEBUG
	std::cout << "Find Edges..." << std::endl;
    #endif
    cv::Mat gray, edge;
    
    { // Local Scope to save memory
	vpp::image2d<vpp::vuchar3> ucharimg(noise_img_.domain());
	
	// Convert double to uchar function
	auto double2uchar = [] (auto& i, const auto& n) {
	    value_type v = n * static_cast<typename MANIFOLD::scalar_type>(std::numeric_limits<unsigned char>::max());
	    vpp::vuchar3 vu = vpp::vuchar3::Zero();
	    vu[0]=static_cast<unsigned char>(v[2]);
	    vu[1]=static_cast<unsigned char>(v[1]);
	    vu[2]=static_cast<unsigned char>(v[0]);
	    i = vu;
	}; 

	vpp::pixel_wise(ucharimg, noise_img_) | double2uchar;
	cv::Mat src = vpp::to_opencv(ucharimg);
	cv::cvtColor(src, gray, CV_BGR2GRAY);
    }

    cv::blur(gray, edge, cv::Size(3,3));
    cv::Canny(edge, edge, 50, 150, 3);
    
    #ifdef TV_DATA_DEBUG
	cv::namedWindow( "Detected Edges", cv::WINDOW_NORMAL ); 
	cv::imshow("Detected Edges", edge);
	cv::waitKey(0);
    #endif

    vpp::image2d<unsigned char> ucharweights = vpp::from_opencv<unsigned char>(edge);
    vpp::pixel_wise(ucharweights, edge_weights_)() | [] (const unsigned char &uw, weights_type& ew) {
	ew = 1.0 - 0.99 * ( static_cast<weights_type>(uw) / static_cast<weights_type>(std::numeric_limits<unsigned char>::max()) );
    };

    #ifdef TV_DATA_DEBUG
	output_weights(edge_weights_, "wedge.csv");
    #endif
}

//TODO: static assert to avoid data that has not exactly 3 channels
// static assert that channel is element {1,2,3}
// make enum RGB out
template < typename MANIFOLD >
void Data<MANIFOLD, 2>::findInpWeights(const int channel){
    #ifdef TV_DATA_DEBUG
	std::cout << "Find Inpainting Area..." << std::endl;
    #endif
    inp_ = inp_mat(noise_img_.domain());
    vpp::pixel_wise(noise_img_, inp_) | [&] (const value_type& img, inp_type& inp) { inp = static_cast<inp_type>(img[channel-1] > 0.95); };
    //vpp::pixel_wise(noise_img_, inp_) | [&] (const value_type& img, inp_type& inp) { inp = static_cast<inp_type>(img[channel-1] > 0.95); };
    #ifdef TV_DATA_DEBUG
	int nr = inp_.nrows();
	int nc = inp_.ncols();

	std::fstream f;
	f.open("inp.csv", std::fstream::out);

	for (int r=0; r<nr; r++){
	    const inp_type* cur = &inp_(r,0);
	    for (int c=0; c<nc; c++){
		f << cur[c];
		if(c != nc-1) f << ",";
	    }
	    f <<  std::endl;
	}
	f.close();
    #endif
}

//TODO: static assert to avoid data that has not exactly 3 channels
// static assert that channel is element {1,2,3}
// make enum RGB out
template < typename MANIFOLD >
void Data<MANIFOLD, 2>::createRandInpWeights(const double threshold){
    
    #ifdef TV_DATA_DEBUG
	std::cout << "Create Random Inp Weights..." << std::endl;
    #endif

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> rand(0.0, 1.0);
 
    inp_ = inp_mat(noise_img_.domain());
    vpp::pixel_wise(noise_img_, inp_) | [&] (const value_type& img, inp_type& inp) { inp = static_cast<inp_type>(rand(gen) > threshold); };
    
    #ifdef TV_DATA_DEBUG
	int nr = inp_.nrows();
	int nc = inp_.ncols();

	std::fstream f;
	f.open("inp.csv", std::fstream::out);

	for (int r=0; r<nr; r++){
	    const inp_type* cur = &inp_(r,0);
	    for (int c=0; c<nc; c++){
		f << cur[c];
		if(c != nc-1) f << ",";
	    }
	    f <<  std::endl;
	}
	f.close();
    #endif
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::rgb_imread(const char* filename){
	static_assert(MANIFOLD::value_dim == 3,"ERROR: RGB Input requires a Manifold with embedding dimension N=3!");
	vpp::image2d<vpp::vuchar3> input_image;
	input_image = vpp::clone(vpp::from_opencv<vpp::vuchar3 >(cv::imread(filename)));
	noise_img_ = storage_type(input_image.domain());
	// Convert Picture of uchar to double 
	    vpp::pixel_wise(input_image, noise_img_) | [] (auto& i, auto& n) {
	    value_type v = value_type::Zero();
	    vpp::vuchar3 vu = i;
	    // TODO: insert manifold scalar type
	    v[0]=static_cast<double>(vu[2]); //opencv saves as BGR
	    v[1]=static_cast<double>(vu[1]);
	    v[2]=static_cast<double>(vu[0]);
	    n = v / static_cast<double>(std::numeric_limits<unsigned char>::max());
	};
    img_ = vpp::clone(noise_img_, vpp::_border = 1);
    //img_ = vpp::clone(noise_img_);

    //TODO: Write separate input functions for weights and inpainting matrices
    weights_ = weights_mat(noise_img_.domain());
    vpp::fill(weights_, 1.0);
    edge_weights_ = vpp::clone(weights_);
    vpp::fill(edge_weights_, 1.0);
    inp_ = inp_mat(noise_img_.domain());
    vpp::fill(inp_, false);
    inpaint_ = false;
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::rgb_readBrightness(const char* filename){
	static_assert(MANIFOLD::value_dim == 1,"ERROR: Brightness Input requires a Manifold with embedding dimension N=3!");
	vpp::image2d<vpp::vuchar3> input_image;
	input_image = vpp::clone(vpp::from_opencv<vpp::vuchar3 >(cv::imread(filename)));
	noise_img_ = storage_type(input_image.domain());
	// Convert Picture of uchar to double 
	    vpp::pixel_wise(input_image, noise_img_)(vpp::_no_threads) | [] (auto& i, auto& n) {
	    vpp::vdouble3 v; 
	    vpp::vuchar3 vu = i;
	    // TODO: insert manifold scalar type
	    v[0]=static_cast<double>(vu[2]); //opencv saves as BGR
	    v[1]=static_cast<double>(vu[1]);
	    v[2]=static_cast<double>(vu[0]);
	    v = v / static_cast<double>(std::numeric_limits<unsigned char>::max());
	    n.setConstant(v.norm()/std::sqrt(3));
	};
    img_ = vpp::clone(noise_img_, vpp::_border = 1);
    //img_ = vpp::clone(noise_img_);

    //TODO: Write separate input functions for weights and inpainting matrices
    weights_ = weights_mat(noise_img_.domain());
    vpp::fill(weights_, 1.0);
    edge_weights_ = vpp::clone(weights_);
    vpp::fill(edge_weights_, 1.0);
    inp_ = inp_mat(noise_img_.domain());
    vpp::fill(inp_, false);
    inpaint_ = false;
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::rgb_readChromaticity(const char* filename){
	static_assert(MANIFOLD::value_dim == 3,"ERROR: Chromaticity Input requires a Manifold with embedding dimension N=3!");
	vpp::image2d<vpp::vuchar3> input_image;
	input_image = vpp::clone(vpp::from_opencv<vpp::vuchar3 >(cv::imread(filename)));
	noise_img_ = storage_type(input_image.domain());
	// Convert Picture of uchar to double 
	    vpp::pixel_wise(input_image, noise_img_) | [] (auto& i, auto& n) {
	    value_type v; 
	    // TODO: insert manifold scalar type
	    v[0]=static_cast<double>(i[2])+1.0; //opencv saves as BGR
	    v[1]=static_cast<double>(i[1])+1.0;
	    v[2]=static_cast<double>(i[0])+1.0;
	    v = v / (static_cast<double>(std::numeric_limits<unsigned char>::max())+1.0);
	    
	    double norm = v.norm();
	    if(norm!=0)
		n = v / norm;
	    else 
		n = v;

	    #ifdef TV_DATA_DEBUG
		if(!std::isfinite(n(0))){
		    std::cout << "\nvuchar pixel " << i << std::endl;
		    std::cout << "double pixel " << v << std::endl; 
		    std::cout << "normalized pixel " << n << std::endl; 
		}
	    #endif
	};
    img_ = vpp::clone(noise_img_, vpp::_border = 1);
    //img_ = vpp::clone(noise_img_);

    //TODO: Write separate input functions for weights and inpainting matrices
    weights_ = weights_mat(noise_img_.domain());
    vpp::fill(weights_, 1.0);
    edge_weights_ = vpp::clone(weights_);
    vpp::fill(edge_weights_, 1.0);
    inp_ = inp_mat(noise_img_.domain());
    vpp::fill(inp_, false);
     inpaint_ = false;
}


template <typename MANIFOLD>
void Data<MANIFOLD, 2>::readMatrixDataFromCSV(const char* filename, const int nx, const int ny){
    #ifdef TV_DATA_DEBUG
	std::cout << "ReadMatrixData from CSV File..." << std::endl;
    #endif
    noise_img_ = storage_type(ny, nx);
    vpp::fill(noise_img_, MANIFOLD::value_type::Zero());

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

    assert(N2==cols);
    assert(nx*ny==rows);
    
    infile.clear();
    infile.seekg(0, std::ios_base::beg);
    
    int i=0;
    Eigen::Matrix<typename MANIFOLD::scalar_type, N2, 1> vectorizedMat;
    vectorizedMat.setZero();

    while (std::getline(infile, line)){
	std::stringstream strstr(line);
	std::string word = "";
	int j=0;
	while (std::getline(strstr, word,',')){
	    typename MANIFOLD::scalar_type entry= static_cast<typename MANIFOLD::scalar_type>(std::stod(word));
	    vectorizedMat(j) = entry;
	    ++j;
	}
	noise_img_(i / nx, i % nx) = Eigen::Map<typename MANIFOLD::value_type>(vectorizedMat.data());
	++i;
    }
    img_ = vpp::clone(noise_img_, vpp::_border = 1);
    fill_border_closest(img_);
    //TODO: Write separate input functions for weights and inpainting matrices
    weights_ = weights_mat(noise_img_.domain());
    vpp::fill(weights_, 1.0);
    edge_weights_ = vpp::clone(weights_);
    vpp::fill(edge_weights_, 1.0);
    inp_ = inp_mat(noise_img_.domain());
    vpp::fill(inp_, false);
    inpaint_ = false;
}

// TODO: Generalize to general N
template <typename MANIFOLD>
void Data<MANIFOLD, 2>::create_nonsmooth_son(const int ny,const int nx){
    #ifdef TV_DATA_DEBUG
	std::cout << "Create Nonsmooth SO(3) Picture..." << std::endl;
    #endif

    const int N = MANIFOLD::value_type::RowsAtCompileTime; 

    static_assert(MANIFOLD::MyType==SO, "ERROR: Only possible for SO(N) manifolds");
    static_assert(N==3, "ERROR: Only possible for SO(N) manifolds");
    
    noise_img_ = storage_type(ny, nx);

    auto son_inserter = [&] (typename MANIFOLD::value_type& v, const vpp::vint2& coord){

	typename MANIFOLD::scalar_type x = (coord(1)-1.0) / nx;
	typename MANIFOLD::scalar_type y = (coord(0)-1.0) / ny;
	Eigen::Matrix<typename MANIFOLD::scalar_type, N, 1> rotation_axis;
	if(x > 0.5)
	    rotation_axis << 2.0 * x, y, 0.0;
	else
	    rotation_axis << 0.0, 2.0 * x, 0.5;
	
	typename MANIFOLD::scalar_type alpha;
	if(x > y)
	    alpha = x + y;
	else
	    alpha = M_PI * 0.5 + x - y;
	
	//v = MANIFOLD::value_type::Identity();
	v = Eigen::AngleAxis<typename MANIFOLD::scalar_type>(alpha, rotation_axis.normalized());
    };

    vpp::pixel_wise(noise_img_, noise_img_.domain()) | son_inserter;
    img_ = vpp::clone(noise_img_, vpp::_border = 1);
    fill_border_closest(img_);

    //TODO: Write separate input functions for weights and inpainting matrices
    weights_ = weights_mat(noise_img_.domain());
    vpp::fill(weights_, 1.0);
    edge_weights_ = vpp::clone(weights_);
    vpp::fill(edge_weights_, 1.0);
    inp_ = inp_mat(noise_img_.domain());
    vpp::fill(inp_, false);
    inpaint_ = false;
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::output_weights(const weights_mat& weights, const char* filename) const{
    int nr = weights.nrows();
    int nc = weights.ncols();

    std::fstream f;
    f.open(filename, std::fstream::out);

    for (int r=0; r<nr; r++){
	const typename Data<MANIFOLD,2>::weights_type* cur = &weights(r,0);
	for (int c=0; c<nc; c++){
	    f << cur[c];
	    if(c != nc-1) f << ",";
	}
	f <<  std::endl;
    }
    f.close();
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::output_matval_img(const char* filename) const{
    int nr = img_.nrows();
    int nc = img_.ncols();

    std::fstream f;
    f.open(filename, std::fstream::out);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "\n");
    for (int r=0; r<nr; r++){
	const auto* cur = &img_(r,0);
	for (int c=0; c<nc; c++)
	    f << cur[c].format(CommaInitFmt);
    }
    f.close();
}
template < typename MANIFOLD >
void Data<MANIFOLD, 2>::output_img(const char* filename) const{
    int nr = img_.nrows();
    int nc = img_.ncols();

    std::fstream f;
    f.open(filename, std::fstream::out);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    for (int r=0; r<nr; r++){
	const auto* cur = &img_(r,0);
	for (int c=0; c<nc; c++){
	    f << cur[c].format(CommaInitFmt);
	    if(c != nc-1) f << ",";
	}
	f <<  std::endl;
    }
    f.close();
}

template < typename MANIFOLD >
void Data<MANIFOLD, 2>::output_nimg(const char* filename) const{
    int nr = noise_img_.nrows();
    int nc = noise_img_.ncols();

    std::fstream f;
    f.open(filename, std::fstream::out);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    for (int r=0; r<nr; r++){
	const auto* cur = &noise_img_(r,0);
	for (int c=0; c<nc; c++){
	    f << cur[c].format(CommaInitFmt);
	    if(c != nc-1) f << ",";
	}
	f <<  std::endl;
    }
    f.close();
}

}// end namespace tvmtl

#endif
