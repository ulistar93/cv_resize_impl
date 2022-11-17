#include <iostream>

#include "opencv2/opencv.hpp"

int main() {
  // image load & crop
  cv::Mat img = cv::imread("../img/input.jpg", CV_8UC1);
  int imgW = img.size().width;
  int imgH = img.size().height;
  int faceSx = 541, faceSy = 199;
  int faceEx = 714, faceEy = 405;
  int roiSx = 368, roiSy = 43;
  int roiEx = 886, roiEy = 561;
  int roiW = roiEx - roiSx;
  int roiH = roiEy - roiSy;
  assert(roiW == roiH);
  // set location
  cv::Rect face(faceSx, faceSy, faceEx - faceSx, faceEy - faceSy);
  cv::Rect roi(roiSx, roiSy, roiW, roiH);
  // crop
  cv::Mat croped_img;
  img(roi).copyTo(croped_img);
  // draw face box & roi box in original image
  cv::rectangle(img, face, cv::Scalar(128));
  cv::rectangle(img, roi, cv::Scalar(255));
  // display
  cv::imshow("input", img);
  cv::imshow("cropped", croped_img);

  /*
  // resizing
  int ksize = 2; // 2 = refer 2 point (INTER_LINEAR)
  const int tW = 320, tH = 320;
  unsigned char* img_roi = croped_img.data;
  unsigned char img320[tW * tH];

  double scale_x = tW / roiW;
  double scale_y = tH / roiH;
  */

  cv::Mat resized_img;
  cv::resize(croped_img, resized_img, cv::Size(320, 320));
  cv::imshow("resized ", resized_img);

  cv::waitKey();
  return 0;
}
