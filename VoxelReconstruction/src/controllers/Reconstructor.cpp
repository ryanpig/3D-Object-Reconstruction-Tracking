/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>
#include <omp.h>
#include "../utilities/General.h"
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2048),
				m_step(32)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(static) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(static) private(v) shared(visible_voxels)
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

	kmean();
}

void writeCSV(string filename, Mat m)
{
	ofstream myfile;
	myfile.open(filename.c_str());
	myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
	myfile.close();
}

/**
* Convert the location (x,y) of each voxel into Point2f 
* Clustering visible voxels by kmean
*/
void Reconstructor::kmean() {
	//m_voxels_amount
	Mat data(m_visible_voxels.size(), 1, CV_32FC2);
	//vector<Point2f> data;
	for (int i = 0; i < m_visible_voxels.size(); i++) {
		Voxel* voxel = m_visible_voxels[i];
		data.at<Point2f>(i,0) = Point2f(voxel->x, voxel->y);
		//data.push_back(Point2f(voxel->x, voxel->y));
	}
	//K-mean 
	//https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
	Mat centers, bestLabels;
	TermCriteria criteria;
	criteria.epsilon = 0.1;
	criteria.maxCount = 1000;
	criteria.type = criteria.EPS;
	int attempts = 3;
	int flags = KMEANS_RANDOM_CENTERS;
	kmeans( data, 4, bestLabels, criteria, attempts, flags, centers);

	//label each voxel 
	for (int j = 0; j < m_visible_voxels.size(); j++) {
		Voxel* voxel = m_visible_voxels[j];
		voxel->group_number = bestLabels.at<int>(j,0);
	}
	//Save the centers
	m_centers.clear();
	for (int k = 0; k < centers.rows; k++) {
		Point2f tmp = Point2f(centers.at<float>(k, 0), centers.at<float>(k, 1));
		m_centers.push_back(tmp);
	}

	//writeCSV("xy.csv", data);
	//writeCSV("label.csv", bestLabels);
	//writeCSV("center.csv", centers);
	//CreateColorModel();

}

/**
* Create color model for each person in each camera
* visible_voxels vector
*/
void Reconstructor::CreateColorModel() {
	//Set the frame to be processed. (Now, the frame is based on when we press key "k")
	
	vector<Mat> allImgs;
	Mat displayImg;
	for (int m = 0; m < m_cameras.size(); m++) {

		//Build BGR matrix by querying pixels, based on grouping number and camera view
		vector<Mat> bgr_planes0, bgr_planes1, bgr_planes2, bgr_planes3;
		QueryPixelsByGroup(0, m, bgr_planes0);  //group 0 , camera m 
		QueryPixelsByGroup(1, m, bgr_planes1);
		QueryPixelsByGroup(2, m, bgr_planes2);
		QueryPixelsByGroup(3, m, bgr_planes3);

		//Calculate ColorHistogram
		vector<Mat> colorModel0, colorModel1, colorModel2, colorModel3;
		GenColorModel(bgr_planes0, colorModel0);
		GenColorModel(bgr_planes1, colorModel1);
		GenColorModel(bgr_planes2, colorModel2);
		GenColorModel(bgr_planes3, colorModel3);

		Mat histImage0, histImage1, histImage2, histImage3;
		// Display Color Histograms
		GenHistogramImg(colorModel0, histImage0);
		GenHistogramImg(colorModel1, histImage1);
		GenHistogramImg(colorModel2, histImage2);
		GenHistogramImg(colorModel3, histImage3);
		Mat LImg, RImg, allImg_tmp,tmp;
		hconcat(histImage0, histImage1, LImg);
		hconcat(histImage2, histImage3, RImg);
		hconcat(LImg, RImg, allImg_tmp);
		allImgs.push_back(allImg_tmp);
	}
		vconcat(allImgs, displayImg);

	namedWindow("Color Histograms", CV_WINDOW_NORMAL);
	imshow("Color Histograms", displayImg);
	waitKey(0);
}

/*
Access BGR of single pixel and add it to vector<uchar>
https://docs.opencv.org/2.4.13.2/doc/user_guide/ug_mat.html
create Mat based on vector<uchar> 
http://answers.opencv.org/question/81831/convert-stdvectordouble-to-mat-and-show-the-image/

*/
//const vector<Mat>&
void Reconstructor::QueryPixelsByGroup(int groupNum, int cameraNo, vector<Mat>& bgr_planes) {
	//vector<Mat> bgr_planes;
	vector<uchar> bvec0, gvec0, rvec0;
	Mat img = m_cameras[cameraNo]->getFrame(); //Get the 2D image from camera x
	for (int j = 0; j < m_visible_voxels.size(); j++) {
		Voxel* voxel = m_visible_voxels[j];
		if (voxel->group_number == groupNum) {
			const Point point = voxel->camera_projection[cameraNo]; 
			//Access BGR of single pixel and add it to matrix												 
			bvec0.push_back(img.at<Vec3b>(point)[0]);
			gvec0.push_back(img.at<Vec3b>(point)[1]);
			rvec0.push_back(img.at<Vec3b>(point)[2]);
		}
	}
	Mat m1, m2, m3;
	m1 = Mat(1, bvec0.size(), CV_8UC1);		bgr_planes.push_back(m1);
	m2 = Mat(1, gvec0.size(), CV_8UC1);		bgr_planes.push_back(m2);
	m3 = Mat(1, rvec0.size(), CV_8UC1);		bgr_planes.push_back(m3);
	memcpy(bgr_planes[0].data, bvec0.data(), bvec0.size() * sizeof(uchar));
	memcpy(bgr_planes[1].data, gvec0.data(), gvec0.size() * sizeof(uchar));
	memcpy(bgr_planes[2].data, rvec0.data(), rvec0.size() * sizeof(uchar));
	//return bgr_planes;
}
/*
Calculate Color Historgram from a given RGB matrix
https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html#histogram-calculation

*/
void Reconstructor::GenColorModel(vector<Mat>& bgr_planes, vector<Mat>& colorModel) {
	// Establish the number of bins
	int histSize = 256;
	// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat b_hist, g_hist, r_hist;
	//vector<Mat> colorModel
	// Compute the histograms:
	calcHist( &(bgr_planes)[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist( &(bgr_planes)[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&(bgr_planes)[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	colorModel.push_back(b_hist);
	colorModel.push_back(g_hist);
	colorModel.push_back(r_hist);

}


void Reconstructor::GenHistogramImg(vector<Mat>& colorModel,Mat& histImageout) {
	Mat b_hist = colorModel[0];
	Mat g_hist = colorModel[1];
	Mat r_hist = colorModel[2];
	//Configuration
	int histSize = 256; 
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	histImageout = histImage;
	/*
	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

	waitKey(0);
	*/
}


} /* namespace nl_uu_science_gmt */
