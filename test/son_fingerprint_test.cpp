#include <iostream>

#include <opencv2/opencv.hpp>

#include <vpp/vpp.hh>
#include <vpp/algorithms/filters/scharr.hh>

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"
#include "../core/visualization.hpp"


void arrowedLine(cv::Mat img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color,
	int thickness=0, int line_type=CV_AA, int shift=0, double tipLength=0.3)
{
    const double tipSize = norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow
    cv::line(img, pt1, pt2, color, thickness, line_type, shift);
    const double angle = std::atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
    cv::Point p(cvRound(pt2.x + tipSize * std::cos(angle + CV_PI / 4)),
    cvRound(pt2.y + tipSize * std::sin(angle + CV_PI / 4)));
    cv::line(img, p, pt2, color, thickness, line_type, shift);
    p.x = cvRound(pt2.x + tipSize * std::cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * std::sin(angle - CV_PI / 4));
    cv::line(img, p, pt2, color, thickness, line_type, shift);
} 



int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	if (argc != 2){
	    std::cerr << "Usage : " << argv[0] << " image" << std::endl;
	    return 1;
	}
	
	std::string fname(argv[1]);

	typedef Manifold< SO, 2 > mf_t;
	typedef Data< mf_t, 2> data_t;	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;
	//typedef algo_traits< mf_t>;
	typedef Visualization<SO, 2, data_t> visual_t;

	vpp::image2d<unsigned char> img = vpp::from_opencv<unsigned char>(cv::imread(argv[1]));
	vpp::image2d<unsigned char> blur_img(img.domain(), vpp::_border = 3);
	cv::GaussianBlur(vpp::to_opencv(img), vpp::to_opencv(blur_img), cv::Size(3,3), 9, 9, cv::BORDER_DEFAULT);

	vpp::image2d<vpp::vdouble2> gradient(img.domain());
	vpp::scharr(blur_img, gradient);	
    
	vpp::image2d<double> angles(img.domain());
	vpp::block_wise(vpp::vint2{5,5}, gradient, angles) | [&] (const auto G, auto A) {
	    double local_block_orientationX = vpp::sum(vpp::pixel_wise(G) | [&] (const auto& g) { return 2.0 * g(1) * g(0); });
	    double local_block_orientationY = vpp::sum(vpp::pixel_wise(G) | [&] (const auto& g) { return g(1) * g(1) - g(0) * g(0); });
	    double a = 0.5 * std::atan2(local_block_orientationX, local_block_orientationY);  
	    vpp::fill(A, a);
	};

	data_t myData = data_t();

	myData.noise_img_ = typename data_t::storage_type(gradient.domain());
	vpp::pixel_wise(angles, myData.noise_img_) | [&] (const auto& a, mf_t::value_type& v){
	    double s = std::sin(a);
	    double c = std::cos(a);
		v << c, -s, s, c;
	};

	myData.img_ = vpp::clone(myData.noise_img_, vpp::_border = 1);
	fill_border_closest(myData.img_);

	myData.initInp();
	myData.initEdgeweights();

	myData.output_matval_img("son_img1.csv");

	cv::Mat original, copy;
	original = cv::imread(argv[1]);
	copy = original.clone();

	double length = 7.0;
	int step = 10;
	
	int ny = angles.nrows();
	int nx = angles.ncols();

	for(int y = 0; y < ny; y += step)
	    for(int x = 0; x < nx; x += step){
		double a =  angles(y,x);
		double cosa = std::cos(a);
		double sina = std::sin(a);

		cv::Point source(x, y);
		cv::Point target(x + length * cosa, y + length * sina);
		arrowedLine(original, source, target, cv::Scalar( 255, 0, 0 ));
	}   

	cv::namedWindow( "Input Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Picture", original);
	cv::imwrite("input_" + fname, original);
	cv::waitKey(0);

	double lam=0.01;
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-10);

	tvmin_t myTVMin(myFunc, myData);

//	std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
//	myTVMin.smoothening(5);

	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();
    
 
	for(int y = 0; y < ny; y += step)
	    for(int x = 0; x < nx; x += step){
		double cosa = myData.img_(y,x)(0,0);
		double cosa2 = cosa*cosa;
		if(cosa2 > 1.0) cosa2 = 1.0;
		double sina = std::sqrt(1.0 - cosa2);

		cv::Point source(x, y);
		cv::Point target(x + length * cosa, y + length * sina);
		arrowedLine(copy, source, target, cv::Scalar( 0, 255, 0 ));
	}

	cv::namedWindow( "Denoised Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Denoised Picture", copy);
	cv::imwrite("denoised_" + fname, copy);
	cv::waitKey(0);



//	visual_t myVisual(myData);

	return 0;
}
