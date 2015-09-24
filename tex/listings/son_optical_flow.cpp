#include <iostream>
#include <string>


#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vpp/vpp.hh>
#include <vpp/algorithms/filters/scharr.hh>

#include "../core/algo_traits.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"
#include "../core/visualization.hpp"

using namespace tvmtl;

void arrowedLine(cv::Mat img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color,
	int thickness=1, int line_type=CV_AA, int shift=0, double tipLength=0.1)
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

void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,
	                    double, const cv::Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step){
	            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
	            arrowedLine(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
	}
}

	typedef Manifold< SO, 2 > mf_t;
	typedef Data< mf_t, 2> data_t;	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;


int main(int argc, const char *argv[])
{

	if (argc != 2 && argc != 3){
	    std::cerr << "Usage : " << argv[0] << "videofile" << " [lambda]" <<std::endl;
	    return 1;
	}
	
	std::string fname(argv[1]);

	cv::VideoCapture cap(fname);
	cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
	cv::Size subPixWinSize(10,10), winSize(31,31);
	const int max_features = 400;


	if( !cap.isOpened() ){
	    std::cout << "Opening of video stream " << fname << " failed!" << std::endl;   
	    return -1;
	}   
   
	int frame_average_window = 5;
	
	cv::Mat prevgray, gray, flow, cflow, frame;
	cv::namedWindow("flow", 1);
	
	int initial_frames_dropped = 50;
	for(int i = 0; i < initial_frames_dropped; ++i  )
	    cap >> frame;

	cap >> frame;
	cvtColor(frame, prevgray, CV_BGR2GRAY);
	cap >> frame;
	cvtColor(frame, gray, CV_BGR2GRAY);
	
	cv::vector<cv::Point2f> features[2];
	cv::goodFeaturesToTrack(prevgray, features[0], max_features, 0.01, 10, cv::Mat(), 3, 0, 0.04);
	cornerSubPix(gray, features[0], subPixWinSize, cv::Size(-1,-1), termcrit);

	cv::goodFeaturesToTrack(gray, features[1], max_features, 0.01, 10, cv::Mat(), 3, 0, 0.04);
	cv::cornerSubPix(gray, features[1], subPixWinSize, cv::Size(-1,-1), termcrit);

	std::vector<unsigned char> status;
	std::vector<float> err;
	cv::calcOpticalFlowPyrLK(prevgray, gray, features[0], features[1], status, err, winSize, 3, termcrit, 0, 0.001);

	cvtColor(prevgray, cflow, CV_GRAY2BGR);
	cv::imwrite("original_optical_flow.jpg", cflow);
	
	for(int i = 0; i < features[1].size(); ++i){
		if( status[i] != 1 )
		    continue;
		arrowedLine(cflow, features[0][i], features[1][i], CV_RGB(255, 0, 0));
	}

	cv::imshow("flow", cflow);
	cv::imwrite("sparse_optical_flow.jpg", cflow);
	cv::waitKey(0);

	int ny = prevgray.rows;
	int nx = prevgray.cols;
	
	data_t myData = data_t();

	myData.noise_img_ = typename data_t::storage_type(ny, nx);
	vpp::fill(myData.noise_img_, mf_t::value_type::Identity());

	myData.initInp();
	vpp::fill(myData.inp_, true);
	myData.inpaint_ = true;

	for(int i = 0; i < features[1].size(); ++i){
		if( status[i] != 1 )
		    continue;

		cv::Point2f pt1 = features[0][i];
		cv::Point2f pt2 = features[1][i];
		double angle = std::atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
		double s = std::sin(angle);
		double c = std::cos(angle);
		myData.noise_img_((int)pt1.y,(int) pt1.x) << c, -s, s, c;
		myData.inp_((int)pt1.y,(int) pt1.x) = false;
	    
	}

	myData.img_ = vpp::clone(myData.noise_img_, vpp::_border = 1);
	fill_border_closest(myData.img_);

	myData.initEdgeweights();

	myData.output_matval_img("son_optical_flow_img.csv");

	double lam=0.01;
	if(argc==3)
	    lam=atof(argv[2]);

	func_t myFunc(lam, myData);
	myFunc.seteps2(1e-12);

	tvmin_t myTVMin(myFunc, myData);

	myTVMin.first_guess();
	//std::cout << "Smoothen picture to obtain initial state for Newton iteration..." << std::endl;
	//myTVMin.smoothening(5);
	myData.output_matval_img("son_optical_flow_img_FIRSTGUESS.csv");

	std::cout << "Start TV minimization..." << std::endl;
	myTVMin.minimize();

	myData.output_matval_img("son_optical_flow_img_FINAL.csv");

	double length = 10.0;
	int step = 18;

	for(int y = 0; y < ny; y += step)
	    for(int x = 0; x < nx; x += step){
		double cosa = myData.img_(y,x)(0,0);
		double cosa2 = cosa*cosa;
		if(cosa2 > 1.0) cosa2 = 1.0;
		double sina = std::sqrt(1.0 - cosa2);

		cv::Point source(x, y);
		cv::Point target(x - length * cosa, y - length * sina);
		arrowedLine(cflow, source, target, cv::Scalar( 0, 255, 0 ));
	}

	cv::imshow("flow", cflow);
	cv::imwrite("reconstructed_optical_flow.jpg", cflow);
	cv::waitKey(0);



	return 0;
}
