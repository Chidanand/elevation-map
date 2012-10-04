#include <iostream>
#include <vector>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

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
  if (argc !=3){
    std::cerr << "Arguments must be passing in this way: ./undistort_image /path/sequence/to/parameters_file.xml /path/sequence/to/image" << std::endl;
    return -1;
  }
  std::string parameters_xml = argv[1];
  std::string image = argv[2];

  /*read the xml file */
  cv::FileStorage fs(parameters_xml, cv::FileStorage::READ);
  
  std::cout << "openning xml file" << std::endl << std::endl;

  if (!fs.isOpened()){
    std::cout << "the xml file coudn't be openned" << std::endl;
  }
  
  // read values
  cv::Mat k, dist;
  fs["P1"] >> k;
  fs["dist1"] >> dist;

  std::cout << "K: " << k << std::endl;
  std::cout << "dist: " << dist << std::endl;

  fs.release();
  
  
  /* Read the image to undistorted */

  cv::Mat mat_im; 
  mat_im  = cv::imread(image, 0);

  if(! mat_im.data ){                             
    std::cerr <<  "Could not open or find the image" << std::endl ;
    return -1;
  }


  /* computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane */
  cv::Mat k_prima, r;
  cv::Size new_image_size = mat_im.size();

  /* Computes the undistortion and rectification transformation map for each camera*/

  cv::Mat map1, map2;
  cv::initUndistortRectifyMap(k, dist, r, k_prima, new_image_size, CV_32FC1, map1, map2);
   
  /* Applies a generic geometrical transformation to an image */

  cv::Mat undistorted_im;

  cv::remap(mat_im, undistorted_im, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
 
  if(! undistorted_im.data ){                     
    std::cerr <<  "Could not open or find the image" << std::endl ;
    return -1;
  }


  // Create Window
  cv::namedWindow("Rectify image", 1);
  cv::imshow("Rectify image", undistorted_im);
  cv::namedWindow("Original image", 1);
  cv::imshow("Original image", mat_im);

  listenToKeys();

  return 0;
}
