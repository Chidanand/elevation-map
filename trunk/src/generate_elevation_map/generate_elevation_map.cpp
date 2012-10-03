#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>



/*Compute the minimun and maximun value from a Mat (Observation: this function can be optimized!) */
void getMinMaxFromMat(cv::Mat_<double>& elevation_map, double& min, double& max){
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

/* change the mat values to the range ...  hay que ecualizar el mapa de altura a valores deseables*/
void changeHeightRange(cv::Mat_<double>& elevation_map, double& min_height, double& max_height){

  for (uint j = 0; j < (uint)elevation_map.cols; j++) {
    for (uint i = 0; i < (uint)elevation_map.rows; i++) {
      if (elevation_map.at<double>(i,j) != 0) // discard disparity null
        elevation_map.at<double>(i,j) += abs(min_height); // min_height is always negative
    }
  }
  return;
}

/*Equalize a mat*/
void matEqualization(cv::Mat_<double>& m, double x1, double y1, double x2, double y2){

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
void printHSV(cv::Mat_<double>& elevation_map, const char* windowName) {

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
  
  // Create Window
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, elevation_map_color);
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
  cv::Mat Q;
  
  fs["Q"] >> Q;
  
  //If size of Q is not 4x4 exit
  if (Q.cols != 4 || Q.rows != 4)
  {
    std::cerr << "ERROR: Could not read matrix Q (doesn't exist or size is not 4x4)" << std::endl;
    return 1;
  }


  //Get the interesting parameters from Q
  double Q03, Q13, Q23, Q32, Q33;
  Q03 = Q.at<double>(0,3);
  Q13 = Q.at<double>(1,3);
  Q23 = Q.at<double>(2,3);
  Q32 = Q.at<double>(3,2);
  Q33 = Q.at<double>(3,3);
  
  std::cout << "Q(0,3) = "<< Q03 <<"; Q(1,3) = "<< Q13 <<"; Q(2,3) = "<< Q23 <<"; Q(3,2) = "<< Q32 <<"; Q(3,3) = "<< Q33 <<";" << std::endl;
  
 
  std::cout << "Read matrix in file " << argv[3] << std::endl;

  //Show the values inside Q (for debug purposes)
  /*
  for (int y = 0; y < Q.rows; y++)
  {
    const double* Qy = Q.ptr<double>(y);
    for (int x = 0; x < Q.cols; x++)
    {
      std::cout << "Q(" << x << "," << y << ") = " << Qy[x] << std::endl;
    }
  }
  */
  
  //Load rgb-image
  cv::Mat img_rgb = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (img_rgb.data == NULL)
  {
    std::cerr << "ERROR: Could not read rgb-image: " << argv[1] << std::endl;
    return 1;
  }
  
  //Load disparity image
  cv::Mat img_disparity = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  
  //equalizate the input image
  //equalizeHist(img_disparity,img_disparity);
  
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
  


  cv::Mat_<double> elevation_map = cv::Mat_<float>(img_rgb.size());
  
  //Get 3d coordinates
  double px, py, pz;
  uchar pr, pg, pb;
 
  for (int i = 0; i < img_rgb.rows; i++)
  {
    uchar* rgb_ptr = img_rgb.ptr<uchar>(i);

    uchar* disp_ptr = img_disparity.ptr<uchar>(i);

    for (int j = 0; j < img_rgb.cols; j++)
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
      
      // std::cout << "Coordenadas pixel: " << "(" << i << ", " << j << ")" << " Height: " << py << std::endl;
      elevation_map.at<double>(i, j) = -py;
    }
  }

  //Show images
  cv::namedWindow("rgb-image");
  cv::namedWindow("disparity-image");

  cv::imshow("rbg-image", img_rgb);
  cv::imshow("disparity-image", img_disparity);
  
  // map height values to the interval [0,max_height+abs(min_height)]
  std::cout << "elevation map value: " << elevation_map.at<double>(400,400) << std::endl;
  
  double min_height, max_height;
  getMinMaxFromMat(elevation_map, min_height, max_height);
  // changeHeightRange(elevation_map, min_height, max_height);
  
  matEqualization(elevation_map, min_height, 0.0, max_height, 100.0);

  std::cout << "elevation map value: " << elevation_map.at<double>(479,639) << std::endl;
  std::cout << "Min height: " << min_height << " Max height: " << max_height << std::endl;
  // std::cout << "elevation_map " << std::endl << elevation_map << std::endl;

  // Show elevation map
  printHSV(elevation_map, "elevation-map");
  
  std::cout << "Press a key to continue..." << std::endl;
  cv::waitKey(0);

  cv::destroyWindow("rgb-image");
  cv::destroyWindow("disparity-image");
  cv::destroyWindow("elevation-map");
  
  return 0;
}
