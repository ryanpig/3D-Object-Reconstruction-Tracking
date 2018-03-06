/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>
#include <opencv2/core/mat.hpp>

#include "Camera.h"
using namespace std;
using namespace cv;


namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */
	struct Voxel
	{
		int x, y, z;                               // Coordinates
		cv::Scalar color;                          // Color
		std::vector<cv::Point> camera_projection;  // Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;  // Flag if camera projection is in camera[c]'s FoV
		int group_number;			   // record group number
	};

private:
	const std::vector<Camera*> &m_cameras;  // vector of pointers to cameras
	const int m_height;                     // Cube half-space height from floor to ceiling
	const int m_step;                       // Step size (space between voxels)

	std::vector<cv::Point3f*> m_corners;    // Cube half-space corner locations

	size_t m_voxels_amount;                 // Voxel count
	cv::Size m_plane_size;                  // Camera FoV plane WxH

	std::vector<Voxel*> m_voxels;           // Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;   // Pointer vector to all visible voxels
	std::vector<vector<Mat>> m_colorModels; // color models offline. 
	std::vector<vector<Mat>> m_colorModels_on; // color models online. 
	std::vector<cv::Point2f> m_centers;     // clustering centers. 
	vector<vector<Point2f>> m_Trajectories; // Trajectories for four clustering centers
	bool m_offline_flag;					// If false, start to build offlice color models.
	bool m_offline_ModelFromFile;			// If true, the offline color models will be retrieved from file. 
	vector<int> m_mapping;					// mapping table to identify which group belongs to which person
	int m_count;		// for export trajectories
	void initialize();

public:
	Reconstructor(
			const std::vector<Camera*> &);
	virtual ~Reconstructor();

	void update();

	//[Clustering & Building color models & Similarity ]
	void kmean();						//clustering voxels 
	void CreateCMsOnline();		     	//Create color model for each person for each view
	void QueryPixelsByGroup(int, int, vector<Mat>&, vector<Mat>&); 
	void GenColorModel(vector<Mat>&, vector<Mat>&); 
	void GenHistogramImg(vector<Mat>, Mat&);	//Generate single 3-ch Histogram image. 
	void CompareColorModels(vector<vector<Mat>>&); // Given color models, it calculate the similarity of each possible pair.
	int CompareColorModels_Online(vector<vector<Mat>>&, vector<Mat>, vector<double>&, bool);
	//[Offline]
	void offlineCMsBuild();				// step by step, to build the offline color models
	void CreateCMsOffline();		    // Loop each view of camera, user has to pick up best color model for each person. The result is saved into a file. 
	void ReadCMsFromFile();				// Read color models from the file. 
	//[Online]
	void onlineCMsBuild();				// step by step, to build the online color models
	void MappingUpdate();				//update group number of each voxel by the mapping table
	void ShowTrajectories();			//Drawing trajectories in an image matrix (file.)

	const vector<int>& getMappingTable() {
		return m_mapping;
	}


	bool getOfflineFlagFromeFile() const {
		return m_offline_ModelFromFile;
	}

	void resetOfflineFlag()
	{
		m_offline_flag = false;
	}

	void setOfflineFlag()
	{
		m_offline_flag = true;
	}

	bool getOfflineFlag() const
	{
		return 	m_offline_flag;
	}

	const std::vector<vector<cv::Point2f>>& getTrajectories() const
	{
		return 	m_Trajectories;
	}

	const std::vector<cv::Point2f>& getClusterCenters() const
	{
		return m_centers;
	}

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
			const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}

};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
