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
	int thickness=1, int line_type=8, int shift=0, double tipLength=0.1)
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
	vpp::pixel_wise(gradient, angles) | [&] (const auto& g, double& a) {
	  a = std::atan2(g(0),g(1));  
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
	myData.initWeights();
	myData.initInp();
	myData.initEdgeWeights();

	myData.output_matval_img("son_img1.csv");

	cv::Mat original, copy;
	original = cv::imread(argv[1]);
	copy = original.clone();

	vpp::image2d<double> angle_with_border = vpp::clone(angles, vpp::_border=5);
	auto BN = vpp::box_nbh2d<double, 5, 5>(angle_with_border);
    
	auto draw_vector =  [&] (const auto& d, const auto& n){
	   int spacing = 12; 
	    if(d(0) % spacing == 0 && d(1) % spacing == 0){
		double avg = 0;
		
		for(int i=-2; i<=2; ++i)
		    for(int j=-2; j<=2; ++j)
			avg +=n(i,j);
		avg/=25.0;
	
		double length = 7.0;

		cv::Point source(d(1),d(0));
		cv::Point target(d(1) + length * std::cos(avg), d(0) + length * std::sin(avg));
		arrowedLine(original, source, target, cv::Scalar( 0, 255, 0 ), 1, 8, 0, 0.35 );
	    }
	};	
	
	
	vpp::pixel_wise(angle_with_border.domain(), BN) | draw_vector;

	cv::namedWindow( "Input Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Picture", original);
	cv::imwrite("input_" + fname, original);
	cv::waitKey(0);

	double lam=0.1;
	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-10);

	tvmin_t myTVMin(myFunc, myData);

	std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
	myTVMin.smoothening(5);

	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();
    
	vpp::pixel_wise(myData.img_, angle_with_border) | [&] (const auto& i, auto& a) {
	    a = std::acos(i(0,0));
	};
 
	original = copy.clone();
	vpp::pixel_wise(angle_with_border.domain(), BN) | draw_vector;

	cv::namedWindow( "Denoised Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Denoised Picture", original);
	cv::imwrite("denoised_" + fname, original);
	cv::waitKey(0);



//	visual_t myVisual(myData);

	return 0;
}
