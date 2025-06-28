// ComputerVision.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4//opencv2/>
#include <iostream>
#include "calib3d.hpp"

#include "imgproc.hpp"

#include <vector>
//debug for d
#ifdef _DEBUG
#pragma comment(lib,"opencv_world4100d")
#else
#pragma comment(lib,"opencv_world4100")
#endif
using namespace cv;
using namespace std;

Mat src,dst;
std::vector<Point2f> points1;
std::vector<Point2f> points2;


void FeatureDescription()
{
	// Step1 Detect features with ORB detector.
	Ptr<ORB> detector = ORB::create();

	std::vector<KeyPoint> keypoints1, keypoints2;

	Mat descriptors1, descriptors2;
	detector->detectAndCompute(src, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(dst, noArray(), keypoints2, descriptors2);
	
	// Step2 Match the feature points.
	BFMatcher matcher(NORM_HAMMING, true);
	vector<DMatch>matches;
	matcher.match(descriptors1, descriptors2, matches);

	// Step3 Filter matches based on distance
	double max_dist = 0, min_dist = 100;
	for (int i = 0; i < matches.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	// Step4 Extract matched points
	
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	Mat img_matches;
	drawMatches(src, keypoints1, dst, keypoints2, matches, img_matches);

	imshow("Matches", img_matches);
	waitKey();

	
	
}

void FindHomographyExample()
{


	Mat ransac = findHomography(points2,points1, RANSAC);
	Size resultSize = Size(dst.cols+src.cols, max(dst.rows,src.rows));
	Mat stitchedImage;
	
	warpPerspective(dst, stitchedImage,ransac , resultSize);
	Mat roi(stitchedImage, Rect(0, 0, dst.cols, dst.rows));
	src.copyTo(roi);

	// Step 5: Display the result
	imshow("Stitched Image(RANSAC)",stitchedImage);
	waitKey(0);


	Mat leastSqaure = findHomography(points2, points1, 0);
	stitchedImage = 0;
	
	warpPerspective(dst, stitchedImage, leastSqaure, resultSize);
	roi = Mat(stitchedImage, Rect(0, 0, dst.cols, dst.rows));
	src.copyTo(roi);

	imshow("Stitched Image(Least Sqaure)", stitchedImage);
	waitKey(0);
}


int main()
{
	src = imread("./Tree1.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		return -1;
	}

    //-- Step 2: Matching descriptor vectors with a brute force matcher
	dst = imread("./Tree2.jpg", IMREAD_COLOR);
	if (dst.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		return -1;
	}
	FeatureDescription();
	FindHomographyExample();

	
	
}
