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
#include <numeric>

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
				m_height(2048), //3076
				m_step(64) //32
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
	//Initialize Trajectories size
	m_Trajectories.resize(4);
	//Initialize once offline color model building process flag.  
	m_offline_flag = false; 
	//Decide if using the color models from file. (True: use CMs from the file "colormodel.txt")
	// By setting false && press key "k" each frame, it trigers a new offline color model building process. 
	m_offline_ModelFromFile = true;
	m_count = 0;

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

/**
 * Build offline Color models and save them into files for further use.
**/

void Reconstructor::offlineCMsBuild() {
	//Clustering 
	kmean();  
	//Build Colod models based on a single frame in different views. 
	//The Color models are stored to file. 
	CreateCMsOffline(); 
	//Read saved color models.
	ReadCMsFromFile();
	//Get metric score among built color models. They should be distiguishable.(low correlation, high ChiSqr)
	CompareColorModels(m_colorModels);
	//End the offline process.
	//m_offline_flag = true;
}

void Reconstructor::onlineCMsBuild() {
	//Clustering 
	kmean();
	//Build Colod models based on a single frame in different views. 
	CreateCMsOnline();
	//Update the group lable of each voxel by mapping result
	MappingUpdate();
	/*
	if (m_count == 10) {
		//Save Trajectories into image
		ShowTrajectories();
		m_count = 0 ;
	}
	m_count++;
	*/
}




void Reconstructor::ReadCMsFromFile() {
	//Read color models back.  
	FileStorage fs;
	fs.open("colormodel.txt", cv::FileStorage::READ);
	if (fs.isOpened())
	{
		vector<Mat> tmp3;
		Mat tmp, tmp2,all;
		Mat histImage0, histImage1, histImage2, histImage3;
		if (m_colorModels.size() != 4) {
			m_colorModels.resize(4);
		}
		fs["CM0"] >> m_colorModels[0];
		fs["CM1"] >> m_colorModels[1];
		fs["CM2"] >> m_colorModels[2];
		fs["CM3"] >> m_colorModels[3];
		GenHistogramImg(m_colorModels[0], histImage0);
		GenHistogramImg(m_colorModels[1], histImage1);
		GenHistogramImg(m_colorModels[2], histImage2);
		GenHistogramImg(m_colorModels[3], histImage3);
		hconcat(histImage0, histImage1, tmp);
		hconcat(histImage2, histImage3, tmp2);
		tmp3.push_back(tmp); tmp3.push_back(tmp2);
		vconcat(tmp3,all);
		imshow("Color Models Read From File", all);
		waitKey(0);
	} 

}

void Reconstructor::CreateCMsOffline() {
	vector<Mat> allImgs, allImgMasks;
	Mat displayImg, displayImgMask;
	vector<Mat> img_masks;
	// Declare what you need
	cv::FileStorage fs("colormodel.txt", cv::FileStorage::WRITE);
	vector<int> modelSelections;
	
	//Each camera View 
	for (int m = 0; m < m_cameras.size(); m++) {
		cout << "Camera View:" << m << endl;
		//Each person
		//Build BGR matrix by querying pixels, based on grouping number
		vector<Mat> bgr_planes0, bgr_planes1, bgr_planes2, bgr_planes3;
		QueryPixelsByGroup(0, m, bgr_planes0, img_masks);  //group 0 , camera m 
		QueryPixelsByGroup(1, m, bgr_planes1, img_masks);
		QueryPixelsByGroup(2, m, bgr_planes2, img_masks);
		QueryPixelsByGroup(3, m, bgr_planes3, img_masks);
		//Projecting clustering voxels onto pixels (img_masks), then showing them by each camera view. 
		Mat LImg_mask, RImg_mask, tmp_mask, tmp_mask_v;
		hconcat(img_masks[0], img_masks[1], LImg_mask);
		hconcat(img_masks[2], img_masks[3], RImg_mask);
		allImgMasks.push_back(LImg_mask);
		allImgMasks.push_back(RImg_mask);
		vconcat(allImgMasks, tmp_mask_v);
		imshow("four masks", tmp_mask_v);
		int key = waitKey(5000);
		//int cvWaitKey(int key);
		//Record how user pick up the best pixel image in each camera view in 5 secs. 
		//Press 1,2,3,4 in each loop (four cameras) to identify the color model represent each person.
		//If we want to pick color models from different frames, we save color models in different files and combine them.
		//Camera 1:Black , Camera 2:Grey, Camera 3:Colorful, Camera 4:Blue
		if (key == 49) {
			modelSelections.push_back(0);
		}else if (key == 50) {
			modelSelections.push_back(1);
		}else if (key == 51) {
			modelSelections.push_back(2);
		}else if (key == 52) {
			modelSelections.push_back(3);
		}else {
			modelSelections.push_back(0);
		}

		img_masks.clear();
		allImgMasks.clear();

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
		Mat LImg, RImg, allImg_tmp, tmp;
		hconcat(histImage0, histImage1, LImg);
		hconcat(histImage2, histImage3, RImg);
		hconcat(LImg, RImg, allImg_tmp);
		allImgs.push_back(allImg_tmp);
		vector<vector<Mat>> CMs_tmp;
		// Save 
		CMs_tmp.push_back(colorModel0);
		CMs_tmp.push_back(colorModel1);
		CMs_tmp.push_back(colorModel2);
		CMs_tmp.push_back(colorModel3);
			//Update selected color model to the file. 
			if (m == 0) {
				m_colorModels.push_back(CMs_tmp[modelSelections[m]]);
				fs << "CM0" << CMs_tmp[modelSelections[m]];
			}
			else if (m == 1) {
				m_colorModels.push_back(CMs_tmp[modelSelections[m]]);
				fs << "CM1" << CMs_tmp[modelSelections[m]];
			}
			else if (m == 2) {
				m_colorModels.push_back(CMs_tmp[modelSelections[m]]);
				fs << "CM2" << CMs_tmp[modelSelections[m]];
			}
			else if (m == 3) {
				m_colorModels.push_back(CMs_tmp[modelSelections[m]]);
				fs << "CM3" << CMs_tmp[modelSelections[m]];
			}

		CMs_tmp.clear();
	}
	//Show 4(camera)*4(clustering) color histograms.  
	vconcat(allImgs, displayImg);
	namedWindow("Color Histograms", CV_WINDOW_NORMAL);
	imshow("Color Histograms", displayImg);
	waitKey(0);
	fs.release();
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
	criteria.maxCount =1000 ;
	criteria.type = criteria.EPS;
	int attempts = 40;
	int flags = KMEANS_RANDOM_CENTERS;
	//loop , check centers, 
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
* 1.Create color model for each person in each camera
* 2.Compare four color models with offline models 
* 
*/
void Reconstructor::CreateCMsOnline() {
	//Set the frame to be processed. (Now, the frame is based on when we press key "k")
	
	vector<Mat> allImgs, allImgMasks;
	Mat displayImg, displayImgMask;
	vector<Mat> img_masks;
	vector<int> compare_result;
	vector<int> duplicate;

	m_mapping.clear();
	vector<double> metrics_person;
	//Each camera View 
	for (int p = 0; p < m_colorModels.size(); p++) {

		compare_result.clear();
		vector<double> metrics;
		for (int m = 0; m < m_cameras.size(); m++) {
			cout << "Camera View:" << m << endl;
			//Build BGR matrix by querying pixels, based on grouping number
			vector<Mat> bgr_planes0, bgr_planes1, bgr_planes2, bgr_planes3;
			QueryPixelsByGroup(0, m, bgr_planes0, img_masks);  //group 0 , camera m 
			QueryPixelsByGroup(1, m, bgr_planes1, img_masks);
			QueryPixelsByGroup(2, m, bgr_planes2, img_masks);
			QueryPixelsByGroup(3, m, bgr_planes3, img_masks);
			allImgMasks.clear();
			//Calculate ColorHistogram
			vector<Mat> colorModel0, colorModel1, colorModel2, colorModel3;
			GenColorModel(bgr_planes0, colorModel0);
			GenColorModel(bgr_planes1, colorModel1);
			GenColorModel(bgr_planes2, colorModel2);
			GenColorModel(bgr_planes3, colorModel3);
			Mat histImage0, histImage1, histImage2, histImage3;
			//(opt) Display Color Histograms 
			
			GenHistogramImg(colorModel0, histImage0);
			GenHistogramImg(colorModel1, histImage1);
			GenHistogramImg(colorModel2, histImage2);
			GenHistogramImg(colorModel3, histImage3);
			Mat LImg, RImg, allImg_tmp, tmp;
			hconcat(histImage0, histImage1, LImg);
			hconcat(histImage2, histImage3, RImg);
			hconcat(LImg, RImg, allImg_tmp);
			allImgs.push_back(allImg_tmp);
			
			//Histogram Comparison
			//Compare four new online color models to one offline color model
			vector<vector<Mat>> colorModels;
			colorModels.push_back(colorModel0);
			colorModels.push_back(colorModel1);
			colorModels.push_back(colorModel2);
			colorModels.push_back(colorModel3);
			// If offline color model is for blue man, we only compare blue channel. 
			int highest_similar;
			if (p == 0) { 
				highest_similar = CompareColorModels_Online(colorModels, m_colorModels[p], metrics, true);
			}else {
				highest_similar = CompareColorModels_Online(colorModels, m_colorModels[p], metrics, false);
			}
			
			compare_result.push_back(highest_similar); //e.g. 0~3
			//build offline version color models
		}
		
		// ---- Color comparison Start ----
		//Vote to decide which color model is highly similar to the offline color model belonging a single person.  
		// Blue -> Grey -> Color -> Black
		
		
		// Find max metric from the highest similarity of each camera view against single offline CM 
		int camNo = distance(metrics.begin(), min_element(metrics.begin(), metrics.end()));
		int groupNo = compare_result[camNo];
		double bestMetric = metrics[camNo];

		cout << "The most similar to the offline model " << p << "is group no" << groupNo << endl;

		int check_exist = count(m_mapping.begin(), m_mapping.end(), groupNo);
		//Save the result to the mapping table
		if (check_exist == 0) {
			m_mapping.push_back(groupNo);
			metrics_person.push_back(bestMetric);
		}
		else {
			//if mapping to same group, compare the best metric to decide the position. 
			ptrdiff_t pos = distance(m_mapping.begin(),find(m_mapping.begin(), m_mapping.end(), groupNo));
			if (metrics_person[pos] <= bestMetric) {
				for (int k = 0; k < 4; k++) { // find available assignment
					int check_ex = count(m_mapping.begin(), m_mapping.end(), k);
					if (check_ex == 0)
					{
						m_mapping.push_back(k);
						metrics_person.push_back(bestMetric);
						break;
					}
				}
			}
			else { //best metric is better than existing metric.
				m_mapping.push_back(groupNo);
				metrics_person.push_back(bestMetric);
				for (int k = 0; k < 4; k++) { // find available 
					int check_ex = count(m_mapping.begin(), m_mapping.end(), k);
					if (check_ex == 0)
					{
						//Reassign group no.  
						m_mapping.at(pos) = k;
						break;
					}
				}

			}
		}

	}
	/*
	vconcat(allImgs, displayImg);
	namedWindow("Color Histograms", CV_WINDOW_NORMAL);
	imshow("Color Histograms", displayImg);
	waitKey(0);
	*/
	

}

/**
 * update voxel label by mapping table. 
 * update trajectories with clustering centers based on mapping table 
**/
void Reconstructor::MappingUpdate() {
	for (int k = 0; k < m_mapping.size(); k++) {
		int mapNo = m_mapping[k]; //get mapped number.
		//update group label of each voxel
		for (int j = 0; j < m_visible_voxels.size(); j++) {
			Voxel* voxel = m_visible_voxels[j];
			if (voxel->group_number == mapNo) {
				voxel->group_number = k;
			}
		}
		//update trajectories
		if (m_offline_flag == true) {
			m_Trajectories[k].push_back(m_centers[mapNo]); 
		}
	}
}

/*
Access BGR of single pixel and add it to vector<uchar>
https://docs.opencv.org/2.4.13.2/doc/user_guide/ug_mat.html
create Mat based on vector<uchar> 
http://answers.opencv.org/question/81831/convert-stdvectordouble-to-mat-and-show-the-image/

*/
void Reconstructor::QueryPixelsByGroup(int groupNum, int cameraNo, vector<Mat>& bgr_planes, vector<Mat>& img_masks) {
	//vector<Mat> bgr_planes;
	vector<uchar> bvec0, gvec0, rvec0;
	Mat img = m_cameras[cameraNo]->getFrame(); //Get the 2D image from camera x
	Mat imgmask(img.rows, img.cols, CV_8UC3, Scalar(0,0 ,0 ));
	for (int j = 0; j < m_visible_voxels.size(); j++) {
		Voxel* voxel = m_visible_voxels[j];
		if (voxel->group_number == groupNum) {
			const Point point = voxel->camera_projection[cameraNo]; 
			//Access BGR of single pixel and add it to matrix												 
			bvec0.push_back(img.at<Vec3b>(point)[0]);
			gvec0.push_back(img.at<Vec3b>(point)[1]);
			rvec0.push_back(img.at<Vec3b>(point)[2]);
			//Save feature image
			imgmask.at<Vec3b>(point) = img.at<Vec3b>(point);


		}
	}
	
	//imshow("mask",imgmask);
	//waitKey(0);

	img_masks.push_back(imgmask);
	Mat m1, m2, m3;
	m1 = Mat(1, bvec0.size(), CV_8UC1);		bgr_planes.push_back(m1);
	m2 = Mat(1, gvec0.size(), CV_8UC1);		bgr_planes.push_back(m2);
	m3 = Mat(1, rvec0.size(), CV_8UC1);		bgr_planes.push_back(m3);
	memcpy(bgr_planes[0].data, bvec0.data(), bvec0.size() * sizeof(uchar));
	memcpy(bgr_planes[1].data, gvec0.data(), gvec0.size() * sizeof(uchar));
	memcpy(bgr_planes[2].data, rvec0.data(), rvec0.size() * sizeof(uchar));
	
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
/*
Based on the color model,the function generates a 3-channel histogram image. 
*/
void Reconstructor::GenHistogramImg(vector<Mat> colorModel,Mat& histImageout) {
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

	// Draw for each channel
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
}

/*
Calculate the similarity between input color models. e.g. 4 CMs -> 6 combinations.
https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
*/

void Reconstructor::CompareColorModels(vector<vector<Mat>>& cms) {
	//Normalization
	for (int k = 0; k < cms.size(); k++) {
		for (int j = 0; j < cms[0].size(); j++) {
			normalize(cms[k][j], cms[k][j], 0, 1, NORM_MINMAX, -1, Mat());
		}
	}

	vector<double> metrics_cor,metrics_chi;
	
	for (int i = 0; i < cms.size(); i++) {
		for (int j = i+1; j < cms.size(); j++) {
				//Correlation
				double metric1_cor = compareHist(cms[i][0], cms[j][0], CV_COMP_CORREL);
				double metric2_cor = compareHist(cms[i][1], cms[j][1], CV_COMP_CORREL);
				double metric3_cor = compareHist(cms[i][2], cms[j][2], CV_COMP_CORREL);
				//ChiSqr
				double metric1_chi = compareHist(cms[i][0], cms[j][0], CV_COMP_CHISQR);
				double metric2_chi = compareHist(cms[i][1], cms[j][1], CV_COMP_CHISQR);
				double metric3_chi = compareHist(cms[i][2], cms[j][2], CV_COMP_CHISQR);
				
				double avg_cor = (metric1_cor + metric2_cor + metric3_cor) / 3;
				double avg_chi = (metric1_chi + metric2_chi + metric3_chi) / 3;
				metrics_cor.push_back(avg_cor);
				metrics_chi.push_back(avg_chi);

				char buffer1[50];
				char buffer2[50];
				int cx1 = snprintf(buffer1, 100, "(%d %d): %.2f , %.2F, %.2f, %.2f", i,j, metric1_cor, metric2_cor, metric3_cor, avg_cor);
				int cx2 = snprintf(buffer2, 100, "(%d %d): %.2f , %.2F, %.2f, %.2f", i, j, metric1_chi, metric2_chi, metric3_chi, avg_chi);

				//cout << "Correlation:" << buffer1 << endl;
			//	cout << "ChiSqr:" << buffer2 << endl;
		
		}
	}
	/*
	cout << "--Correlation--" << endl;
	cout << "H similarity:" << distance(metrics_cor.begin(), max_element(metrics_cor.begin(), metrics_cor.end())) << endl;
	cout << "L similarity:" << distance(metrics_cor.begin(), min_element(metrics_cor.begin(), metrics_cor.end())) << endl;
	cout << "--ChiSqr--" << endl;
	cout << "H similarity:" << distance(metrics_chi.begin(), min_element(metrics_chi.begin(), metrics_chi.end())) << endl;
	cout << "L similarity:" << distance(metrics_chi.begin(), max_element(metrics_chi.begin(), metrics_chi.end())) << endl;
	*/
}
/**
* Compare online four models v.s. single offline model 
* Input: four online CMs, one offline CM
* Output: highest index of CMs. 
**/

int Reconstructor::CompareColorModels_Online(vector<vector<Mat>>& cms, vector<Mat> cmoff, vector<double>& metrics,bool onlyBlue) {
	//Normalization
	for (int k = 0; k < cms.size(); k++) {
		for (int j = 0; j < cms[0].size(); j++) {
			normalize(cms[k][j], cms[k][j], 0, 1, NORM_MINMAX, -1, Mat());
		}
	}
	for (int r = 0; r < cmoff.size(); r++) {
		normalize(cmoff[r], cmoff[r], 0, 1, NORM_MINMAX, -1, Mat());
	}

	vector<double> metrics_cor, metrics_chi;

	for (int i = 0; i < cms.size(); i++) {
			//Correlation
			double metric1_cor = compareHist(cms[i][0], cmoff[0], CV_COMP_CORREL);
			double metric2_cor = compareHist(cms[i][1], cmoff[1], CV_COMP_CORREL);
			double metric3_cor = compareHist(cms[i][2], cmoff[2], CV_COMP_CORREL);
			//ChiSqr
			double metric1_chi = compareHist(cms[i][0], cmoff[0], CV_COMP_CHISQR);
			double metric2_chi = compareHist(cms[i][1], cmoff[0], CV_COMP_CHISQR);
			double metric3_chi = compareHist(cms[i][2], cmoff[0], CV_COMP_CHISQR);

			double avg_cor = (metric1_cor + metric2_cor + metric3_cor) / 3;
			double avg_chi = (metric1_chi + metric2_chi + metric3_chi) / 3;
			//Only compare Blue channel for Blue guy
			if (onlyBlue == true) { 
				avg_cor = metric1_cor;
				avg_chi = metric1_chi;
			}

			metrics_cor.push_back(avg_cor);
			metrics_chi.push_back(avg_chi);

			char buffer1[50];
			char buffer2[50];
			int cx1 = snprintf(buffer1, 100, "(%d ): %.2f , %.2F, %.2f, %.2f", i, metric1_cor, metric2_cor, metric3_cor, avg_cor);
			int cx2 = snprintf(buffer2, 100, "(%d ): %.2f , %.2F, %.2f, %.2f", i, metric1_chi, metric2_chi, metric3_chi, avg_chi);

		//	cout << "Correlation:" << buffer1 << endl;
		//	cout << "ChiSqr:" << buffer2 << endl;

	}
	int hcor, hsqr, lcor, lsqr;
	hcor = distance(metrics_cor.begin(), max_element(metrics_cor.begin(), metrics_cor.end()));
	lcor = distance(metrics_cor.begin(), min_element(metrics_cor.begin(), metrics_cor.end()));
	hsqr = distance(metrics_chi.begin(), max_element(metrics_chi.begin(), metrics_chi.end()));
	lsqr = distance(metrics_chi.begin(), min_element(metrics_chi.begin(), metrics_chi.end()));
	/*
	cout << "--Correlation--" << endl;
	cout << "H similarity:" << hcor << endl;
	cout << "L similarity:" << lcor << endl;
	//Remember lsqr means high similarity
	cout << "--ChiSqr--" << endl;
	cout << "L similarity:" << hsqr << endl;
	cout << "H similarity:" << lsqr << endl;
	*/
	metrics.push_back(metrics_chi[lsqr]);
	// Though we build both correlation & chisur, but now we only use chisqr.
	return lsqr; 

}

void Reconstructor::ShowTrajectories() {
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	Mat img(m_height*2, m_height*2, CV_8UC3, Scalar(255,255 ,255));
	
	for (int p = 0; p < m_Trajectories.size(); p++) {
		for (int k = 0; k < m_Trajectories[p].size(); k++) {
			Point2f pt = m_Trajectories[p][k];
			pt.x = int(pt.x + m_height);
			pt.y = int(pt.y + m_height);
			Vec3b colorB (255,0,0);
			Vec3b colorG(0,255, 0);
			Vec3b colorR(0, 0, 255);
			Vec3b colorGr(40, 40, 40);
				if (p == 0) {
					img.at<Vec3b>(pt)[0] = 255;
				}else if (p == 1) {
					img.at<Vec3b>(pt)[1] = 255;
				}else if (p == 2) {
					img.at<Vec3b>(pt)[2] = 255;
				}else if (p == 3) {
					img.at<Vec3b>(pt) = colorGr;
				}
			}
		}
	
	//Save image

	namedWindow("trajectory", CV_WINDOW_KEEPRATIO);
	imshow("trajectory", img);
	waitKey(10);
	imwrite("trajectories.png", img);

}



/** Offline CMs construction



/** ATTACKING Plan
* (Online) Process each single frame
* 1. Get the current visible voxels (m_visible)
* 2. K-mean for clustering -> label, centers 
* 3. Build color models by labels 
* 4. Mapping by comparing (offline) color models
* 5. Store the centers in grouping vector to form trajectories (opt:with Frame Number)(Vector<Point2f> Tra1,Tra2,..)
* 6. Draw m_Trajectories in the Image by different color

**/




} /* namespace nl_uu_science_gmt */
