#include <iostream>

#include "opencv2/opencv.hpp"

int main() {
  cv::Mat img = cv::imread("../img/input.jpg");
  cv::imshow("input", img);

  const int charImgW = 1280;
  const int charImgH = 720;
  int imgW = img.size().width;
  int imgH = img.size().height;
  assert(imgW == charImgW || imgH == charImgH);
  char charimg[charImgW * charImgH];

  cv::waitKey();
  return 0;
}
