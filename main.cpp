// COMP 499: Computer Vision, Project
// Robert Nguyen (ID# 21697048)

#include <iostream>
#include <algorithm>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"

void harrisCornerResponse(cv::Mat & src, std::string windowName, int threshold, std::vector<cv::KeyPoint> & keypoints);
bool isLocalMax(cv::Point & p, cv::Mat & src);
float getAngle(float x_dir, float y_dir);
void project(int x1, int y1, cv::Mat& H, double& x2, double& y2);
int computeInlierCount(cv::Mat& H, std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, int& inlierThreshold);
void getInliers(cv::Mat& hom, std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, int& inlierThreshold, std::vector<cv::DMatch> & inliers);
void ransac(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, int& numIterations, int& inlierThreshold, cv::Mat & hom, cv::Mat & homInv, cv::Mat img1, cv::Mat& img2);
void stitch(cv::Mat& img1, cv::Mat& img2, cv::Mat& hom, cv::Mat& homInv, cv::Mat& stitchedImg);

int main(int argc, char** argv)
{
	std::string filename0, filename1, filename2;
	int threshold0, threshold1, threshold2;
	std::vector<cv::KeyPoint> img0_keypoints = std::vector<cv::KeyPoint>();
	std::vector<cv::KeyPoint> img1_keypoints = std::vector<cv::KeyPoint>();
	std::vector<cv::KeyPoint> img2_keypoints = std::vector<cv::KeyPoint>();

	int numIterations;
	int inlierThreshold;

	std::cout << "Enter image 0 filename (no stitching; only corner detection): ";
	std::cin >> filename0;

	std::cout << "Enter image 0 harris corner threshold: ";
	std::cin >> threshold0;

	std::cout << "Enter image 1 filename (stitch with image 2): ";
	std::cin >> filename1;

	std::cout << "Enter image 1 harris corner threshold: ";
	std::cin >> threshold1;

	std::cout << "Enter image 2 filename (to be stitched with image 1): ";
	std::cin >> filename2;

	std::cout << "Enter image 2 harris corner threshold: ";
	std::cin >> threshold2;

	std::cout << "Enter # of RANSAC iterations:";
	std::cin >> numIterations;

	std::cout << "Enter inlier threshold (RANSAC):";
	std::cin >> inlierThreshold;

	std::string input_image0 = "_img/" + filename0;
	std::string input_image = "_img/" + filename1;
	std::string input_image2 = "_img/" + filename2;

	cv::Mat src0 = cv::imread(input_image0, cv::IMREAD_COLOR);
	cv::Mat src1 = cv::imread(input_image, cv::IMREAD_COLOR);
	cv::Mat src2 = cv::imread(input_image2, cv::IMREAD_COLOR);

	if (src0.empty() || src1.empty() || src2.empty())
	{
		std::cout << "An image could not be opened." << std::endl;
		return -1;
	}
	else
	{
		std::cout << "Images loaded!" << std::endl;
		std::cout << "Image 0: ";
		std::cout << "Rows: " << src0.rows << ", Cols: " << src0.cols << std::endl;
		std::cout << "Image 1: ";
		std::cout << "Rows: " << src1.rows << ", Cols: " << src1.cols << std::endl;
		std::cout << "Image 2: ";
		std::cout << "Rows: " << src2.rows << ", Cols: " << src2.cols << std::endl;
	}

	cv::Mat src0_gray, src1_gray, src2_gray;
	cv::cvtColor(src0, src0_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(src1, src1_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(src2, src2_gray, cv::COLOR_BGR2GRAY);

	harrisCornerResponse(src0_gray, "img1_", threshold0, img0_keypoints);
	harrisCornerResponse(src1_gray, "img1_", threshold1, img1_keypoints);
	harrisCornerResponse(src2_gray, "img2_", threshold2, img2_keypoints);
	cv::Mat harris_out0, harris_out1, harris_out2;
	cv::drawKeypoints(src0, img0_keypoints, harris_out0, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
	cv::drawKeypoints(src1, img1_keypoints, harris_out1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
	cv::drawKeypoints(src2, img2_keypoints, harris_out2, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
	cv::imshow("harris0", harris_out0);
	cv::imshow("harris1", harris_out1);
	cv::imshow("harris2", harris_out2);
	cv::imwrite("_out/1a.png", harris_out0);
	cv::imwrite("_out/1b.png", harris_out1);
	cv::imwrite("_out/1c.png", harris_out2);
	cv::waitKey(0);

	cv::Ptr<cv::Feature2D> f2d = cv::SIFT::create();
	cv::Mat descriptors_1, descriptors_2;
	f2d->compute(src1, img1_keypoints, descriptors_1);
	f2d->compute(src2, img2_keypoints, descriptors_2);

	cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_L2SQR, true);
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	std::cout << "Matches: " << matches.size() << std::endl;

	cv::Mat match_out;
	cv::drawMatches(src1, img1_keypoints, src2, img2_keypoints, matches, match_out);
	cv::imshow("matchy", match_out);
	cv::imwrite("_out/2.png", match_out);
	cv::waitKey(0);

	cv::Mat hom, homInv, stitchedImg;
	ransac(matches, img1_keypoints, img2_keypoints, numIterations, inlierThreshold, hom, homInv, src1, src2);
	stitch(src1, src2, hom, homInv, stitchedImg);
	cv::waitKey(0);

	cv::destroyAllWindows();
}

void project(int x1, int y1, cv::Mat& H, double& x2, double& y2) {
	double u =	((double)x1 * H.at<double>(0, 0)) +	((double)y1 * H.at<double>(0, 1)) + (H.at<double>(0, 2));
	double v =	((double)x1 * H.at<double>(1, 0)) +	((double)y1 * H.at<double>(1, 1)) + (H.at<double>(1, 2));
	double w =	((double)x1 * H.at<double>(2, 0)) +	((double)y1 * H.at<double>(2, 1)) + (H.at<double>(2, 2));
	x2 = (u / w);
	y2 = (v / w);

	/*
	if ((x1 == 375) && (y1 == 1020)) {
		std::cout << "u: " << u << std::endl;
		std::cout << "v: " << v << std::endl;
		std::cout << "w: " << w << std::endl;
		std::cout << "x2 (u/w): " << x2 << std::endl;
		std::cout << "y2 (v/w): " << y2 << std::endl;
	}*/
}

int computeInlierCount(cv::Mat& H, std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, int& inlierThreshold) {
	int inlierCount = 0;
	float x1, y1;
	double x2, y2;
	int kpt1_match_index, kpt2_match_index;
	float act_x2, act_y2;

	for (int i = 0; i < matches.size(); i++) {
		kpt1_match_index = matches[i].queryIdx;
		kpt2_match_index = matches[i].trainIdx;
		x1 = kpts1[kpt1_match_index].pt.x;
		y1 = kpts1[kpt1_match_index].pt.y;
		project(x1, y1, H, x2, y2);

		// actual x,y
		act_x2 = kpts2[kpt2_match_index].pt.x;
		act_y2 = kpts2[kpt2_match_index].pt.y;

		// calculate dist between (x1,y1) and (act_x2, act_y2)
		double dist_x = x2 - act_x2;
		double dist_y = y2 - act_y2;
		dist_x *= dist_x;
		dist_y *= dist_y;
		double distance = cv::sqrt(dist_x + dist_y);

		if (distance < (double)inlierThreshold) {
			inlierCount++;
		}
	}
	return inlierCount;
}

void getInliers(cv::Mat& hom, std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, int& inlierThreshold, std::vector<cv::DMatch> & inliers) {
	float x1, y1;
	double x2, y2;
	int kpt1_match_index, kpt2_match_index;
	float act_x2, act_y2;

	for (int i = 0; i < matches.size(); i++) {
		kpt1_match_index = matches[i].queryIdx;
		kpt2_match_index = matches[i].trainIdx;
		x1 = kpts1[kpt1_match_index].pt.x;
		y1 = kpts1[kpt1_match_index].pt.y;
		project(x1, y1, hom, x2, y2);

		act_x2 = kpts2[kpt2_match_index].pt.x;
		act_y2 = kpts2[kpt2_match_index].pt.y;

		// calculate dist between (x1,y1) and (act_x2, act_y2)
		cv::Point2d dist = cv::Point2d(x2, y2) - cv::Point2d(act_x2, act_y2);
		double distance = cv::sqrt(dist.x*dist.x + dist.y*dist.y);

		if (distance < (double)inlierThreshold) {
			inliers.push_back(matches[i]);
		}
	}
}

void ransac(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, int& numIterations, int& inlierThreshold, cv::Mat & hom, cv::Mat & homInv, cv::Mat img1, cv::Mat& img2) {
	cv::Mat cur_best_homography;
	int cur_highest_homography_inliers = 0;

	for (int i = 0; i < numIterations; i++) {
		if (i % 10000 == 0) {
			std::cout << "Progress: " << ((float)i/(float)numIterations*100.0) << "%" << std::endl;
		}

		// take 4 random pairs, generate 4 random indexes
		int pt_pair_1, pt_pair_2, pt_pair_3, pt_pair_4;
		pt_pair_1 = rand() % matches.size();
		pt_pair_2 = rand() % matches.size();
		while (pt_pair_2 == pt_pair_1) {
			pt_pair_2 = rand() % matches.size();
		}
		pt_pair_3 = rand() % matches.size();
		while (pt_pair_3 == pt_pair_1 || pt_pair_3 == pt_pair_2) {
			pt_pair_3 = rand() % matches.size();
		}
		pt_pair_4 = rand() % matches.size();
		while (pt_pair_4 == pt_pair_1 || pt_pair_4 == pt_pair_2 || pt_pair_4 == pt_pair_3) {
			pt_pair_4 = rand() % matches.size();
		}
		int im1_ind1 = matches[pt_pair_1].queryIdx;
		int im1_ind2 = matches[pt_pair_2].queryIdx;
		int im1_ind3 = matches[pt_pair_3].queryIdx;
		int im1_ind4 = matches[pt_pair_4].queryIdx;

		int im2_ind1 = matches[pt_pair_1].trainIdx;
		int im2_ind2 = matches[pt_pair_2].trainIdx;
		int im2_ind3 = matches[pt_pair_3].trainIdx;
		int im2_ind4 = matches[pt_pair_4].trainIdx;

		cv::Point2f im1_pt1 = kpts1[im1_ind1].pt;
		cv::Point2f im1_pt2 = kpts1[im1_ind2].pt;
		cv::Point2f im1_pt3 = kpts1[im1_ind3].pt;
		cv::Point2f im1_pt4 = kpts1[im1_ind4].pt;

		cv::Point2f im2_pt1 = kpts2[im2_ind1].pt;
		cv::Point2f im2_pt2 = kpts2[im2_ind2].pt;
		cv::Point2f im2_pt3 = kpts2[im2_ind3].pt;
		cv::Point2f im2_pt4 = kpts2[im2_ind4].pt;

		//prepare for feeding into cv::findHomography
		std::vector<cv::Point2f> img1_pts, img2_pts;

		img1_pts.push_back(cv::Point2f(im1_pt1.x, im1_pt1.y));
		img1_pts.push_back(cv::Point2f(im1_pt2.x, im1_pt2.y));
		img1_pts.push_back(cv::Point2f(im1_pt3.x, im1_pt3.y));
		img1_pts.push_back(cv::Point2f(im1_pt4.x, im1_pt4.y));

		img2_pts.push_back(cv::Point2f(im2_pt1.x, im2_pt1.y));
		img2_pts.push_back(cv::Point2f(im2_pt2.x, im2_pt2.y));
		img2_pts.push_back(cv::Point2f(im2_pt3.x, im2_pt3.y));
		img2_pts.push_back(cv::Point2f(im2_pt4.x, im2_pt4.y));

		cv::Mat H = cv::findHomography(img1_pts, img2_pts, 0);

		int cur_homography_inliers = computeInlierCount(H, matches, kpts1, kpts2, inlierThreshold);
		if (cur_homography_inliers > cur_highest_homography_inliers) {
			cur_best_homography = H;
			cur_highest_homography_inliers = cur_homography_inliers;
			std::cout << "New highest inliers: " << cur_highest_homography_inliers << ", On iter #" << (i+1) << std::endl;
			std::cout << "Homography: \n" << cur_best_homography << std::endl;
			std::cout << "********" << std::endl;
		}
	}
	std::cout << "Highest # inliers for homography: " << cur_highest_homography_inliers << std::endl;
	std::cout << "Best homography: \n" << cur_best_homography << std::endl;
	std::cout << "********" << std::endl;

	std::vector<cv::DMatch> inliers;
	getInliers(cur_best_homography, matches, kpts1, kpts2, inlierThreshold, inliers);

	// draw inlier matches
	cv::Mat out;
	cv::drawMatches(img1, kpts1, img2, kpts2, inliers, out);
	cv::imshow("ransac", out);
	cv::imwrite("_out/3.png", out);
	cv::waitKey(0);

	std::vector<cv::Point2f> pts1, pts2;

	for (int ind = 0; ind < inliers.size(); ind++) {
		pts1.push_back(kpts1[inliers[ind].queryIdx].pt);
		pts2.push_back(kpts2[inliers[ind].trainIdx].pt);
	}

	hom = cv::findHomography(pts1, pts2, 0);
	cv::invert(hom, homInv);
	homInv /= homInv.at<double>(2, 2);
	std::cout << "Refined homography: \n" << hom << std::endl;
	std::cout << "Inverse homography: \n" << homInv << std::endl;
}

void stitch(cv::Mat& img1, cv::Mat& img2, cv::Mat& hom, cv::Mat& homInv, cv::Mat& stitchedImg) {
	// compute size of stitchedImage
	int x_ll = 0;
	int y_ll = img2.rows-1;
	int x_ul = 0;
	int y_ul = 0;
	int x_lr = img2.cols-1;
	int y_lr = img2.rows-1;
	int x_ur = img2.cols-1;
	int y_ur = 0;

	double x_ll_2, y_ll_2,
		x_ul_2, y_ul_2,
		x_lr_2, y_lr_2,
		x_ur_2, y_ur_2;
	
	// project img2 extrema with inverse homography
	project(x_ll, y_ll, homInv, x_ll_2, y_ll_2);
	project(x_ul, y_ul, homInv, x_ul_2, y_ul_2);
	project(x_lr, y_lr, homInv, x_lr_2, y_lr_2);
	project(x_ur, y_ur, homInv, x_ur_2, y_ur_2);

	std::vector<double> all_x = { x_ll_2, x_ul_2, x_lr_2, x_ur_2 };
	std::vector<double> all_y = { y_ll_2, y_ul_2 , y_lr_2 , y_ur_2 };

	auto [min_x, max_x] = std::minmax_element(all_x.begin(), all_x.end());
	auto [min_y, max_y] = std::minmax_element(all_y.begin(), all_y.end());

	double max_width = *max_x;
	double min_width = *min_x;
	double max_height = *max_y;
	double min_height = *min_y;

	// offset in case any part of img2 goes on top or to the left of img1
	int offset_x_stitch = std::abs(std::min(0, (int)std::floor(min_width)));
	int offset_y_stitch = std::abs(std::min(0, (int)std::floor(min_height)));

	int width_stitched = offset_x_stitch + std::max(img1.cols, (int)std::ceil(max_width));
	int height_stitched = offset_y_stitch + std::max(img1.rows, (int)std::ceil(max_height));

	/*
	std::cout << "max x: " << *max_x << std::endl;
	std::cout << "min x: " << *min_x << std::endl;
	std::cout << "max y: " << *max_y << std::endl;
	std::cout << "min y: " << *min_y << std::endl;
	std::cout << "width: " << width_stitched << ", height:" << height_stitched << std::endl;
	std::cout << "UL (before & after): (" << x_ul << "," << y_ul << ") -> (" << x_ul_2 << "," << y_ul_2 << ")" << std::endl;
	std::cout << "UR (before & after): (" << x_ur << "," << y_ur << ") -> (" << x_ur_2 << "," << y_ur_2 << ")" << std::endl;
	std::cout << "LL (before & after): (" << x_ll << "," << y_ll << ") -> (" << x_ll_2 << "," << y_ll_2 << ")" << std::endl;
	std::cout << "LR (before & after): (" << x_lr << "," << y_lr << ") -> (" << x_lr_2 << "," << y_lr_2 << ")" << std::endl;
	std::cout << "offset_x_stitch: " << offset_x_stitch << std::endl;
	std::cout << "offset_y_stitch: " << offset_y_stitch << std::endl;
	*/

	stitchedImg = cv::Mat::zeros(cv::Size(width_stitched, height_stitched), CV_32FC3);

	// copy image1 onto stitchedImage
	img1.convertTo(img1, CV_32FC3, 1/255.0);
	stitchedImg.convertTo(stitchedImg, CV_32FC3, 1 / 255.0);

	for (int y = 0; y < img1.rows; y++) {
		for (int x = 0; x < img1.cols; x++) {
			stitchedImg.at<cv::Vec3f>(y + offset_y_stitch, x + offset_x_stitch) = img1.at<cv::Vec3f>(y, x);
		}
	}
	// project every pixel of stitchedImage into image2
	for (int y = 0; y < stitchedImg.rows; y++) {
		for (int x = 0; x < stitchedImg.cols; x++) {
			double x2, y2;

			// project cur (x,y) point from stitchedImg onto img2 (minus offset)
			project(x-offset_x_stitch, y-offset_y_stitch, hom, x2, y2);

			// is it within boundary?
			if ((x2 <= img2.cols-1) && (x2 >= 0) && (y2 <= img2.rows-1) && (y2 >= 0)) {

				cv::Mat patch = cv::Mat::zeros(cv::Size(1, 1), CV_32FC3);

				cv::getRectSubPix(img2, cv::Size(1, 1), cv::Point2d(x2, y2), patch, CV_32FC3);
				patch.convertTo(patch, CV_32FC3, 1 / 255.0);

				stitchedImg.at<cv::Vec3f>(y, x) = patch.at<cv::Vec3f>(0, 0);
			}
		}
	}

	cv::imshow("stitched", stitchedImg);
	stitchedImg.convertTo(stitchedImg, CV_8UC3, 255.0);
	cv::imwrite("_out/4.png", stitchedImg);
}

void harrisCornerResponse(cv::Mat & src, std::string window_name, int threshold, std::vector<cv::KeyPoint> & img_keypoints)
{

	cv::Mat x2y2, xy, mtrace2, x_derivative, y_derivative, x2_derivative, y2_derivative, xy_derivative, x2g_derivative, y2g_derivative, xyg_derivative, r, r_norm;

	// compute gradients
	// Ix
	cv::Sobel(src, x_derivative, CV_32FC1, 1, 0, 3, cv::BORDER_DEFAULT);
	// Iy
	cv::Sobel(src, y_derivative, CV_32FC1, 0, 1, 3, cv::BORDER_DEFAULT);
	// Ix^2
	cv::pow(x_derivative, 2.0, x2_derivative);
	// Iy^2
	cv::pow(y_derivative, 2.0, y2_derivative);
	// IxIy
	cv::multiply(x_derivative, y_derivative, xy_derivative);

	// gaussian mask (5x5)
	cv::Size mask_size = cv::Size(5, 5);
	cv::GaussianBlur(x2_derivative, x2g_derivative, mask_size, 1.0, 0.0, cv::BORDER_DEFAULT);
	cv::GaussianBlur(y2_derivative, y2g_derivative, mask_size, 0.0, 1.0, cv::BORDER_DEFAULT);
	cv::GaussianBlur(xy_derivative, xyg_derivative, mask_size, 1.0, 1.0, cv::BORDER_DEFAULT);

	/*
	M =	| x^2g_d	xyg_d	|
		| xyg_d		y^2g_d	|
	*/

	cv::multiply(x2g_derivative, y2g_derivative, x2y2);
	cv::multiply(xyg_derivative, xyg_derivative, xy);
	cv::pow((x2g_derivative + y2g_derivative), 2.0, mtrace2);

	// R = det(M) - alpha*trace(M)^2
	float alpha = 0.04;
	r = (x2y2 - xy) - alpha * mtrace2;

	cv::normalize(r, r_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	int pts_of_interest = 0;
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			float harrisResponse = r_norm.at<float>(j, i);
			if ((int)harrisResponse > threshold)
			{
				cv::Point cur_pt = cv::Point(i, j);

				if (isLocalMax(cur_pt, r_norm))	// non-max suppression
				//if (true)
				{
					pts_of_interest++;

					float x_dir = x_derivative.at<float>(j, i);
					float y_dir = y_derivative.at<float>(j, i);
					float keyPoint_angle = getAngle(x_dir, y_dir);

					// shouldn't be seeing this
					if ((keyPoint_angle > 360) || (keyPoint_angle < 0))
						std::cout << keyPoint_angle << "!";

					img_keypoints.push_back(cv::KeyPoint(cur_pt.x, cur_pt.y, 5, keyPoint_angle, harrisResponse, 0, -1));
				}
			}
		}
	}
	std::cout << "Pts of interest for " + window_name + ": " << pts_of_interest << std::endl;
}

float getAngle(float x_dir, float y_dir)
{
	float resp = atan2(-y_dir, x_dir) * 180 / 3.141592 + 180;
	return std::clamp(resp, 0.0f, 360.0f);
}

// returns whether or not the pt is a local 3x3 neighbourhood (current DoG image only)
bool isLocalMax(cv::Point & p, cv::Mat & src)
{
	// is there anything to the left?
	if (p.x - 1 > -1)
	{
		// centre-left
		if ((src.at<float>(p.y, p.x - 1) > src.at<float>(p)))
			return false;

		// bottom-left
		if (p.y - 1 > -1)
			if ((src.at<float>(p.y - 1, p.x - 1) > src.at<float>(p)))
				return false;

		// top-left
		if (p.y + 1 < src.rows)
			if ((src.at<float>(p.y + 1, p.x - 1) > src.at<float>(p)))
				return false;
	}

	// is there anything to the right?
	if (p.x + 1 < src.cols)
	{
		// centre-right
		if ((src.at<float>(p.y, p.x + 1) > src.at<float>(p)))
			return false;

		// bottom-right
		if (p.y - 1 > -1)
			if ((src.at<float>(p.y - 1, p.x + 1) > src.at<float>(p)))
				return false;

		// top-right
		if (p.y + 1 < src.rows)
			if ((src.at<float>(p.y + 1, p.x + 1) > src.at<float>(p)))
				return false;
	}

	// top-centre
	if (p.y + 1 < src.rows)
		if ((src.at<float>(p.y + 1, p.x) > src.at<float>(p)))
			return false;

	// bottom-centre
	if (p.y - 1 > -1)
		if ((src.at<float>(p.y - 1, p.x) > src.at<float>(p)))
			return false;

	return true;
}