#include <iostream>

#include <opencv2/opencv.hpp>

#include <vpp/vpp.hh>
#include <vpp/algorithms/filters/scharr.hh>

#include <mtvmtl/core/algo_traits.hpp>
#include <mtvmtl/core/data.hpp>
#include <mtvmtl/core/functional.hpp>
#include <mtvmtl/core/tvmin.hpp>
#include <mtvmtl/core/visualization.hpp>


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


void thinningIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
            for (int j = 1; j < im.cols-1; j++)
            {
	                uchar p2 = im.at<uchar>(i-1, j);
	                uchar p3 = im.at<uchar>(i-1, j+1);
	                uchar p4 = im.at<uchar>(i, j+1);
	                uchar p5 = im.at<uchar>(i+1, j+1);
	                uchar p6 = im.at<uchar>(i+1, j);
	                uchar p7 = im.at<uchar>(i+1, j-1);
	                uchar p8 = im.at<uchar>(i, j-1);
	                uchar p9 = im.at<uchar>(i-1, j-1);
	    
	                int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
	                         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
	                         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
	                         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
	                int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
	                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
	                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
	    
	                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
	                    marker.at<uchar>(i,j) = 1;
	            }
        }

    im &= ~marker;
}


void thinning(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
            thinningIteration(im, 0);
            thinningIteration(im, 1);
            cv::absdiff(im, prev, diff);
            im.copyTo(prev);
        } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}



int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	if (argc != 2 && argc!= 3){
	    std::cerr << "Usage : " << argv[0] << " image" << " [lambda]" << std::endl;
	    return 1;
	}
	
	std::string fname(argv[1]);

	typedef Manifold< SO, 2 > mf_t;
	typedef Data< mf_t, 2> data_t;	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;
	//typedef algo_traits< mf_t>;
	typedef Visualization<SO, 2, data_t> visual_t;

	cv::Mat src = cv::imread(argv[1]);
	    if (src.empty())
		        return -1;

        cv::Mat bw;
        cv::cvtColor(src, bw, CV_BGR2GRAY);
        cv::threshold(bw, bw, 10, 255, CV_THRESH_BINARY);

        thinning(bw);
	cv::imshow("dst", bw);
	cv::imwrite("skeleton_" + fname, bw);

	vpp::image2d<unsigned char> img = vpp::from_opencv<unsigned char>(bw);
	//vpp::image2d<unsigned char> blur_img(img.domain(), vpp::_border = 3);
	//cv::GaussianBlur(vpp::to_opencv(img), vpp::to_opencv(blur_img), cv::Size(3,3), 9, 9, cv::BORDER_DEFAULT);

	vpp::image2d<vpp::vdouble2> gradient(img.domain());
	//vpp::scharr(blur_img, gradient);	
	vpp::scharr(img, gradient);	
    
	vpp::image2d<double> angles(img.domain());
	vpp::block_wise(vpp::vint2{5,5}, gradient, angles) | [&] (const auto G, auto A) {
	    double local_block_orientationX = vpp::sum(vpp::pixel_wise(G) | [&] (const auto& g) { return 2.0 * g(1) * g(0); });
	    double local_block_orientationY = vpp::sum(vpp::pixel_wise(G) | [&] (const auto& g) { return g(1) * g(1) - g(0) * g(0); });
	    double a = 0.5 * std::atan2(local_block_orientationX, local_block_orientationY);
	    a += 0.5 * M_PI;
	    vpp::fill(A, a);
	};
	data_t myData = data_t();
	myData.noise_img_ = typename data_t::storage_type(gradient.domain());
	vpp::pixel_wise(angles, myData.noise_img_) | [&] (const auto& a, mf_t::value_type& v){
	    double s = std::sin(a);
	    double c = std::cos(a);
		v << c, -s, s, c;
	};
/*	
	data_t myData = data_t();
	myData.noise_img_ = typename data_t::storage_type(gradient.domain());
	myData.initInp();
	myData.inpaint_ = true;
	vpp::fill(myData.inp_, true);

	vpp::image2d<double> angles(img.domain());

	vpp::pixel_wise(gradient, myData.noise_img_, myData.inp_, angles) | [&] (const auto& g, mf_t::value_type& i, bool inp, double a){
	    if(g.norm()>1e4){
	    inp = false;
	    a = std::atan2(g(0),g(1));
	    double s = std::sin(a);
	    double c = std::cos(a);
		i << c, -s, s, c;
	    }
	    else
		a=0;
	};
*/
	myData.img_ = vpp::clone(myData.noise_img_, vpp::_border = 1);
	fill_border_closest(myData.img_);

	myData.initInp();
	myData.initEdgeweights();

	myData.output_matval_img("son_img1.csv");

	cv::Mat original, copy;
	original = cv::imread(argv[1]);
	copy = original.clone();

	double length = 10.0;
	int step = 15;
	
	int ny = angles.nrows();
	int nx = angles.ncols();

	for(int y = 0; y < ny; y += step)
	    for(int x = 0; x < nx; x += step){
		double a =  angles(y,x);
		double cosa = std::cos(a);
		double sina = std::sin(a);

		cv::Point source(x, y);
		cv::Point target(x + length * cosa, y + length * sina);
		arrowedLine(original, source, target, cv::Scalar( 0, 0, 255 ));
	}   

	cv::namedWindow( "Input Picture", cv::WINDOW_NORMAL ); 
	cv::imshow( "Input Picture", original);
	cv::imwrite("input_" + fname, original);
	cv::waitKey(0);

	double lam=1.5;
	if(argc==3)
	    lam=atof(argv[2]);

	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-16);

	tvmin_t myTVMin(myFunc, myData);

//	myTVMin.first_guess();

	std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
	myTVMin.smoothening(5);

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
