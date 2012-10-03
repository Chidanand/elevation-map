#include <iostream>
#include <vector>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "../libraries/libelas/src/elas.h"

/// Global Variables
int trackbar_max;
int alpha_slider;
double alpha_track_bar;
double beta_track_bar;

cv::Mat undistorted_left_im, undistorted_right_im, dst, undistorted_right_im_moved;
cv::Mat_<float> D1_data, D2_data;

//function to show disparity map with HSV color

void printHSV(cv::Mat_<float>& disparityData, const char* windowName) {
  cv::Mat_<cv::Vec3b> disparity_data_color(disparityData.size());
  for (uint j = 0; j < (uint)disparityData.cols; j++) {
    for (uint i = 0; i < (uint)disparityData.rows; i++) {    
      cv::Vec3b v;
      
      float val = std::min(disparityData.at<float>(i,j) * 0.01f, 1.0f);
      if (val <= 0) {
        v[0] = v[1] = v[2] = 0;
      } else {
        float h2 = 6.0f * (1.0f - val);
        unsigned char x  = (unsigned char)((1.0f - fabs(fmod(h2, 2.0f) - 1.0f))*255);
        if (0 <= h2&&h2<1) { v[0] = 255; v[1] = x; v[2] = 0; }
        else if (1 <= h2 && h2 < 2)  { v[0] = x; v[1] = 255; v[2] = 0; }
        else if (2 <= h2 && h2 < 3)  { v[0] = 0; v[1] = 255; v[2] = x; }
        else if (3 <= h2 && h2 < 4)  { v[0] = 0; v[1] = x; v[2] = 255; }
        else if (4 <= h2 && h2 < 5)  { v[0] = x; v[1] = 0; v[2] = 255; }
        else if (5 <= h2 && h2 <= 6) { v[0] = 255; v[1] = 0; v[2] = x; }
      }
      
      disparity_data_color.at<cv::Vec3b>(i, j) = v;
    }
  }
  
  // Create Window
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, disparity_data_color);
}

// Function which calculate disparity (both maps) and show them using the function printHSV

void showDisparity() {
  Elas::parameters param;
  param.postprocess_only_left = true; //true = better precition, false = more fast
  param.ipol_gap_width = 3; // parammeter to interpolate pixles and get a better look of the image
  
  /*Call the constructor of class elas to compute disparity maps */
  Elas elas(param);

  cv::Mat_<uchar> im1_out_gray, im2_out_gray;
  
  undistorted_left_im.copyTo(im2_out_gray);
  undistorted_right_im_moved.copyTo(im1_out_gray);

  // get image width and height
  int32_t width  = im1_out_gray.size().width;
  int32_t height = im1_out_gray.size().height;

  // allocate memory for disparity images
  const int32_t dims[3] = {width,height,width}; // bytes per line = width
  
  D1_data = cv::Mat_<float>(im1_out_gray.size());
  D2_data = cv::Mat_<float>(im1_out_gray.size());

  // process
  elas.process(im2_out_gray.data, im1_out_gray.data, (float*)D1_data.data, (float*)D2_data.data, dims);

  // show disparity on windows openned
  printHSV(D1_data, "Disparity Right Camera");
  printHSV(D2_data, "Disparity Left Camera");
}

/**
 * @function on_trackbar
 * @brief Callback for trackbar
 */
void on_trackbar(int, void*){
  int height = undistorted_right_im.size().height;
  int width = undistorted_right_im.size().width;
  double left_transparency = 0.5;
  double right_transparency = 0.5;
  int x_prima, d_prima;
  
  undistorted_right_im.copyTo(undistorted_right_im_moved);
  alpha_slider -= width;
  
  for(int y = 0; y<height; ++y){
    for(int x = 0; x<width; ++x){
      if (alpha_slider < 0)
      {
        if (x < width + alpha_slider)
        {
          d_prima = (width + alpha_slider - x);
          x_prima = width - d_prima;
          undistorted_right_im_moved.at<uchar>(y,x) = undistorted_right_im.at<uchar>(y,x_prima);
        }
        else
        {
          undistorted_right_im_moved.at<uchar>(y,x) = 0;
        }
      }
      else 
      {
        if (x + alpha_slider < width)
        {
          x_prima = alpha_slider +x;
          undistorted_right_im_moved.at<uchar>(y,x_prima) = undistorted_right_im.at<uchar>(y,x);
        }
        if (x < alpha_slider){
          undistorted_right_im_moved.at<uchar>(y,x) = 0;
        }
          
        }
      }
    }
  
  cv::addWeighted(undistorted_left_im, left_transparency, undistorted_right_im_moved, right_transparency, 0.0, dst);

  cv::imshow( "Linear Blend", dst);
  std::cout << "Alpha slider: " << alpha_slider << std::endl;
  showDisparity();
}


// Funtion to capture keys from keyboard
void listenToKeys() {
  int keyPressed;
  bool exit = false;

  while(!exit) {
    keyPressed = cv::waitKey(0);

    switch(keyPressed) {
    case ' ': //space
    case 1048608: //space (with numlock on)
      exit = true;
      break;
    }
  }
}



int main(int argc, char *argv[])
{
  if (argc !=4){
    std::cerr << "Arguments must be passing in this way: ./generate_disparity_image /path/sequence/to/parameters_file.xml /path/sequence/to/left_image /path/sequence/to/right_image" << std::endl;
    return -1;
  }
  std::string parameters_xml = argv[1];
  std::string left_image = argv[2];
  std::string right_image = argv[3];

  /*read the xml file */
  cv::FileStorage fs(parameters_xml, cv::FileStorage::READ);
  
  std::cout << "openning xml file" << std::endl << std::endl;

  if (!fs.isOpened()){
    std::cout << "the xml file coudn't be openned" << std::endl;
  }
  
  // read values
  cv::Mat left_k, right_k, left_dist, right_dist, r, t;
  fs["P1"] >> left_k;
  fs["P2"] >> right_k;
  fs["dist1"] >> left_dist;
  fs["dist2"] >> right_dist;
  fs["R"] >> r;
  fs["T"] >> t;
  
  fs.release();
  
  
  /* Read the image to undistorted */

  cv::Mat mat_left_im, mat_right_im; 
  mat_left_im  = cv::imread(left_image, 0);
  mat_right_im  = cv::imread(right_image, 0);

  /* computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane */
  cv::Mat left_r, right_r, left_k_prima, right_k_prima, q;
  double alpha = 0;
  cv::Size new_left_image_size = mat_left_im.size();
  cv::Size new_right_image_size = mat_right_im.size();
  cv::Rect left_roi, right_roi;

  cv::stereoRectify(left_k, left_dist, right_k, right_dist, mat_left_im.size(), r, t, left_r, right_r, left_k_prima, right_k_prima, q, alpha, cv::CALIB_ZERO_DISPARITY, new_left_image_size , &left_roi , &right_roi );


  /* Computes the undistortion and rectification transformation map for each camera*/

  cv::Mat left_map1, left_map2, right_map1, right_map2;
  cv::initUndistortRectifyMap(left_k, left_dist, left_r, left_k_prima, new_left_image_size, CV_32FC1, left_map1, left_map2);
  cv::initUndistortRectifyMap(right_k, right_dist, right_r, right_k_prima, new_right_image_size, CV_32FC1, right_map1, right_map2);
  
  /* Applies a generic geometrical transformation to an image */

  cv::remap(mat_left_im, undistorted_left_im, left_map1, left_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
  cv::remap(mat_right_im, undistorted_right_im, right_map1, right_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);


  if(! mat_left_im.data ){                             
    std::cerr <<  "Could not open or find the left image" << std::endl ;
    return -1;
  }
  
  if(! mat_right_im.data ){                            
    std::cerr <<  "Could not open or find the right image" << std::endl ;
    return -1;
  }

  if(! undistorted_left_im.data ){                     
    std::cerr <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  if(! undistorted_right_im.data ){                    
    std::cerr <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  // Initialize values
  alpha_slider = -(undistorted_left_im.size().width-1);
  trackbar_max = 2 * ( undistorted_left_im.size().width-1);
  undistorted_right_im_moved = cv::Mat(undistorted_right_im.size(),CV_8UC1);
  
  // Create Window
  cv::namedWindow("Linear Blend", 1);
 
  // Create Trackbars
  char trackbarName[50];
  sprintf(trackbarName, "dst [0,%d]", trackbar_max);
  
  cv::createTrackbar(trackbarName, "Linear Blend", &alpha_slider, trackbar_max, on_trackbar);

  cv::setTrackbarPos(trackbarName, "Linear Blend", 632);

  cv::Mat dst;

  // call on_trackbar function to align left and right images, then calculate and show disparity maps.
  on_trackbar(alpha_slider, 0);

  listenToKeys();

  /* Save Disparity left image, this image will be used to create the 3D-Point-Cloud */
  std::vector<int> qualityType;
  qualityType.push_back(CV_IMWRITE_PXM_BINARY);
  qualityType.push_back(1);
  cv::imwrite("left_disparity.pgm", D2_data, qualityType);

  /*Save reprojection matrix Q in a XML file*/
  std::cout << "Matriz Q: " << q << std::endl;

  cv::FileStorage fs_out("Q.xml", cv::FileStorage::WRITE);
  fs_out << "Q" << q;

  fs_out.release();
  
  return 0;
}
