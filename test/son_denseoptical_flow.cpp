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

void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,
	                    double, const cv::Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step){
	            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
	            arrowedLine(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
	}
}

int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	if (argc != 2){
	    std::cerr << "Usage : " << argv[0] << "videofile" << std::endl;
	    return 1;
	}
	
	std::string fname(argv[1]);

	typedef Manifold< SO, 2 > mf_t;
	typedef Data< mf_t, 2> data_t;	
	typedef Functional<FIRSTORDER, ISO, mf_t, data_t> func_t;
	typedef TV_Minimizer< IRLS, func_t, mf_t, data_t, OMP > tvmin_t;

	cv::VideoCapture cap(fname);

	if( !cap.isOpened() ){
	    std::cout << "Opening of video stream " << fname << " failed!" << std::endl;   
	    return -1;
	}   
   
	int frame_average_window = 5;
	
	cv::Mat prevgray, gray, flow, avgflow, cflow, frame;
	cv::namedWindow("flow", 1);
	/*   
	std::cout << "Average Optical Flow over " << frame_average_window << " frames..." << std::endl;
	for(int i = 0; i < frame_average_window + 1; ++i) {
	    cap >> frame;
	    cvtColor(frame, gray, CV_BGR2GRAY);

	    if( prevgray.data ){
		cv::calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		if( avgflow.data )
		    cv::accumulate(flow, avgflow);
		else
		    avgflow = flow;
	    }
	    std::swap(prevgray, gray);
	}

        avgflow /= frame_average_window;
	std::cout << "Averaging done." << std::endl;
	
	cvtColor(avgflow, cflow, CV_GRAY2BGR);
        drawOptFlowMap(avgflow, cflow, 16, 1.5, CV_RGB(0, 255, 0));
	*/
	int initial_frames_dropped = 50;
	for(int i = 0; i < initial_frames_dropped; ++i  )
	    cap >> frame;

	cap >> frame;
	cvtColor(frame, prevgray, CV_BGR2GRAY);
	cap >> frame;
	cvtColor(frame, gray, CV_BGR2GRAY);
		
	cv::calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	cvtColor(prevgray, cflow, CV_GRAY2BGR);
        drawOptFlowMap(flow, cflow, 16, 1.5, CV_RGB(0, 255, 0));

	cv::imshow("flow", cflow);
	cv::waitKey(0);
	
	return 0;
}
