#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>

int angle_slider;
double pi = 3.14159265;
cv::Mat img_disparity, Q;
cv::Mat_<double> elevation_map;


/*Compute the minimun and maximun value from a Mat (Observation: this function can be optimized!) */
void getMinMaxFromMat(cv::Mat& elevation_map, double& min, double& max){
  double min_value = 0.0, max_value = 0.0;
  
  for (uint j = 0; j < (uint)elevation_map.cols; j++) {
    for (uint i = 0; i < (uint)elevation_map.rows; i++) {
  
      if (min_value > elevation_map.at<double>(i, j)) {
        min_value = elevation_map.at<double>(i, j);
      }
      
      if (max_value < elevation_map.at<double>(i, j)) {
        max_value = elevation_map.at<double>(i, j);
      }
    }
  }
  
  min = min_value;
  max = max_value;
  return;
}


/*Equalize a mat*/
void matEqualization(cv::Mat& m, double x1, double y1, double x2, double y2){

  // compute the line f(x) = a*x + b between the points (x1,y1) (x2,y2)
  double dx, dy, a, b;

  dx = x2 - x1;
  dy = y2 - y1;
  
  a = dy / dx;
  b = y1 - a * x1;

  //linear equalization
  for (uint j = 0; j < (uint)m.cols; j++) {
    for (uint i = 0; i < (uint)m.rows; i++) {
      if (m.at<double>(i,j) != 0) // discard disparity null
        m.at<double>(i,j) = a * m.at<double>(i,j) + b;
    }
  }
  return;
}


//function to show disparity map with HSV color
void printHSV(cv::Mat& elevation_map, const char* windowName) {

  cv::Mat_<cv::Vec3b> elevation_map_color(elevation_map.size());
  
  for (uint j = 0; j < (uint)elevation_map.cols; j++) {
    for (uint i = 0; i < (uint)elevation_map.rows; i++) {
      
      cv::Vec3b v;
      double val = std::min(elevation_map.at<double>(i,j) * 0.01d, 1.0d);
      if (val <= 0) {
        v[0] = v[1] = v[2] = 0;
      } else {
        double h2 = 6.0d * (1.0d - val);
        unsigned char x  = (unsigned char)((1.0f - fabs(fmod(h2, 2.0f) - 1.0f))*255);
        if (0 <= h2&&h2<1) { v[0] = 255; v[1] = x; v[2] = 0; }
        else if (1 <= h2 && h2 < 2)  { v[0] = x; v[1] = 255; v[2] = 0; }
        else if (2 <= h2 && h2 < 3)  { v[0] = 0; v[1] = 255; v[2] = x; }
        else if (3 <= h2 && h2 < 4)  { v[0] = 0; v[1] = x; v[2] = 255; }
        else if (4 <= h2 && h2 < 5)  { v[0] = x; v[1] = 0; v[2] = 255; }
        else if (5 <= h2 && h2 <= 6) { v[0] = 255; v[1] = 0; v[2] = x; }
      }
      
      elevation_map_color.at<cv::Vec3b>(i, j) = v;
    }
  }
  
  // show elevation map
  cv::imshow(windowName, elevation_map_color);
}

/*compute elevation map using reprojection matrix and the correct Rotation of the camera respect to the real world coordinates (parallel to the floor, for example)
@ R is the rotation matrix of the camera respect to the real world
@ elevation_map is the output elevation map
*/
void computeElevationMap(cv::Mat& disparity_map, cv::Mat& Q, cv::Mat& R, cv::Mat& elevation_map){

  //If size of Q is not 4x4 exit
  if (Q.cols != 4 || Q.rows != 4)
  {
    std::cerr << "ERROR: Could not read matrix Q (doesn't exist or size is not 4x4)" << std::endl;
    return;
  }

  //Get the interesting parameters from Q
  double Q03, Q13, Q23, Q32, Q33;
  Q03 = Q.at<double>(0,3);
  Q13 = Q.at<double>(1,3);
  Q23 = Q.at<double>(2,3);
  Q32 = Q.at<double>(3,2);
  Q33 = Q.at<double>(3,3);
  
  //Get 3d coordinates
  double px, py, pz;
  uchar pr, pg, pb;

  cv::Mat_<double> v = cv::Mat(2,1,CV_32FC1);
  
  for (int i = 0; i < disparity_map.rows; i++)
  {
    uchar* rgb_ptr = disparity_map.ptr<uchar>(i);

    uchar* disp_ptr = disparity_map.ptr<uchar>(i);

    for (int j = 0; j < disparity_map.cols; j++)
    {

      uchar d = disp_ptr[j];
      if ( d == 0 ) continue; //Discard bad pixels
      double pw = -1.0 * static_cast<double>(d) * Q32 + Q33; 
      px = static_cast<double>(j) + Q03;
      py = static_cast<double>(i) + Q13;
      pz = Q23;
      
      px = px/pw;
      py = py/pw;
      pz = pz/pw;
      
      ////////////////////////////////////
      v.at<double>(0,0) = pz;
      v.at<double>(1,0) = py;
      v = R * v;
      
      pz = v.at<double>(0,0);
      py = v.at<double>(1,0);
      
      //std::cout << " (x,y,z) :   " << "(" << px << "," << py << "," << pz << ")" << std::endl;
      ////////////////////////////////////

      // std::cout << "Coordenadas pixel: " << "(" << i << ", " << j << ")" << " Height: " << py << std::endl;
      elevation_map.at<double>(i, j) = -py;
    }
  }

  return;
}


/**
 * @function on_trackbar
 * @brief Callback for trackbar
 */
void on_trackbar(int, void*){

  /*calculate rotation matrix R*/
  double theta = angle_slider * pi/180;

  cv::Mat_<double> R = cv::Mat(2,2,CV_32FC1);
  R.at<double>(0,0)= cos(-theta);
  R.at<double>(0,1)= sin(-theta);
  R.at<double>(1,0)= -sin(-theta);
  R.at<double>(1,1)= cos(-theta);

  computeElevationMap(img_disparity, Q, R, elevation_map);

  cv::imshow( "elevation-map", angle_slider);
  std::cout << "angle slider: " << angle_slider << std::endl;
  
  // Show elevation map
  double min_height, max_height;
  getMinMaxFromMat(elevation_map, min_height, max_height);
  
  /* 0 is the lowest and 100.0 is the higest value that can take the printHSV function */
  matEqualization(elevation_map, min_height, 0.0, max_height, 100.0);

  printHSV(elevation_map, "elevation-map");
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


int main( int argc, char** argv )
{
  //Check arguments
  if (argc != 4)
  {
    std::cerr << "Usage: " << argv[0] << " <rgb-image-filename> <disparity-image-filename> <path-to-Q-matrix>" << std::endl;
    return 1;
  }

  //Load Matrix Q
  cv::FileStorage fs(argv[3], cv::FileStorage::READ);
 
  fs["Q"] >> Q;
  
  //If size of Q is not 4x4 exit
  if (Q.cols != 4 || Q.rows != 4)
  {
    std::cerr << "ERROR: Could not read matrix Q (doesn't exist or size is not 4x4)" << std::endl;
    return 1;
  }
  
  //Load rgb-image
  cv::Mat img_rgb = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (img_rgb.data == NULL)
  {
    std::cerr << "ERROR: Could not read rgb-image: " << argv[1] << std::endl;
    return 1;
  }
  
  //Load disparity image
  img_disparity = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  
  if (img_disparity.data == NULL)
  {
    std::cerr << "ERROR: Could not read disparity-image: " << argv[2] << std::endl;
    return 1;
  }
  
  //Both images must be same size
  if (img_rgb.size() != img_disparity.size())
  {
    std::cerr << "ERROR: rgb-image and disparity-image have different sizes " << std::endl;
    return 1;
  }

  elevation_map = cv::Mat(img_rgb.size(),CV_32FC1);

  
  //Show images
  cv::namedWindow("rgb-image");
  cv::namedWindow("disparity-image");

  // Create Trackbars
  char trackbarName[50];
  int trackbar_max = 359;
  angle_slider = 0;
  sprintf(trackbarName, "Angle [0,%d]", trackbar_max);
  cv::namedWindow("elevation-map", 1);
  cv::createTrackbar(trackbarName, "elevation-map", &angle_slider, trackbar_max, on_trackbar);

  cv::setTrackbarPos(trackbarName, "elevation-map", 0);

  on_trackbar(angle_slider, 0);

  cv::imshow("rgb-image", img_rgb);
  cv::imshow("disparity-image", img_disparity);
     
  std::cout << "Press a bar space to continue..." << std::endl;
  listenToKeys();

  cv::destroyWindow("rgb-image");
  cv::destroyWindow("disparity-image");
  cv::destroyWindow("elevation-map");
  
  return 0;
}
