#include <iostream>
#include <math.h>

#include "opencv2/opencv.hpp"

int main() {
  // image load & crop
  cv::Mat img = cv::imread("../img/input.jpg", CV_8UC1);
  int imgW = img.size().width;
  int imgH = img.size().height;
  int faceSx = 541, faceSy = 199;
  int faceEx = 714, faceEy = 405;
  int roiSx = 368, roiEx = 886, roiSy = 43, roiEy = 561; // 518 x 518
  //int roiSx = 467, roiEx = 787, roiSy = 142, roiEy = 462; // 320 x 320
  //int roiSx = 467, roiEx = 786, roiSy = 142, roiEy = 461; // 319 x 319
  //int roiSx = 467, roiEx = 788, roiSy = 142, roiEy = 463; // 321 x 321
  //int roiSx = 612, roiEx = 644, roiSy = 286, roiEy = 318; // 32 x 32
  //int roiSx = 612, roiEx = 617, roiSy = 286, roiEy = 291; // 5 x 5
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

//#define CV_DEBUG
  
#ifdef CV_DEBUG
  cv::Mat resized_img;
  cv::resize(croped_img, resized_img, cv::Size(320, 320));
#else
  // resizing impl
  // src
  int sW = roiW, sH = roiH;
  int ssize = sW * sH;
  unsigned char* src = croped_img.data;
  // dst
  const int dW = 320, dH = 320;
  int dsize = dW * dH;
  //unsigned char dst[dW * dH] = { 0 };
  unsigned char dst[102400] = { 0 };
  // const
  const int INTER_RESIZE_COEF_BITS = 11;
  const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

  double inv_scale_x = (double) dW / sW;
  double inv_scale_y = (double) dH / sH;
  double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;

  int k, sx, sy, dx, dy;
  int cn = 1; //channel number
  float fx, fy;
  const int ksize = 2; // 2 = refer 2 point (INTER_LINEAR)
  int xmin = 0, xmax = dW, width = dW * cn;

  unsigned char* _buffer = (unsigned char*)malloc((width + dH) * (sizeof(int) + sizeof(short) * ksize));
  //memset(_buffer, 0, (width + dH) * (sizeof(int) + sizeof(short) * ksize));
  int* xofs = (int*)_buffer;
  int* yofs = xofs + width;
  short* alpha = (short*)(yofs + dH);
  short* beta = alpha + width*ksize;
  float cbuf[ksize] = {0};

  for (dx = 0; dx < dW; dx++) {
      fx = (float)((dx + 0.5) * scale_x - 0.5);
	  sx = (int)floor(fx);
	  fx -= sx;

      if (sx < 0) {
          xmin = dx + 1; // xmin don't use later, since src point refer S0[xofs[dx]] and S0[xofs[dx]+1] only
          fx = 0, sx = 0;
      }
      if (sx >= sW - 1) {
          xmax = std::min(xmax, dx);
          fx = 0, sx = sW - 1;
      }

	  for (k = 0, sx *= cn; k < cn; k++)
        xofs[dx * cn + k] = sx + k;
      //xofs[dx] = sx; // cn=1
	  cbuf[0] = 1.f - fx;
	  cbuf[1] = fx;
	  for (k = 0; k < ksize; k++)
		  alpha[dx * cn * ksize + k] = static_cast<short>(round(cbuf[k] * INTER_RESIZE_COEF_SCALE));
  }
  // for (dy = 0; dy < dH; dy++) should be same above
  // Because src and dst image is also square size 518x518 -> 320x320
  memcpy(yofs, xofs, sizeof(int)*width);
  memcpy(beta, alpha, sizeof(short)*width*ksize);

  int* row_buffer = (int*)malloc(sizeof(int) * dW * ksize); // 320*2
  //memset(row_buffer, 0, sizeof(int) * dW * ksize);
  int* rows[ksize] = { 0 };
  unsigned char* srows[ksize] = { 0 };
  int prev_sy[ksize];
  for (int k = 0; k < ksize; k++)
  {
      prev_sy[k] = -1;
      rows[k] = row_buffer + dW * k;
  }


  for (dy = 0; dy < dH; dy++) {
	  int sy0 = yofs[dy], k0 = ksize, k1 = 0, ksize2 = ksize / 2;
      for (int k = 0; k < ksize; k++)
      {
          int sy = sy0 - ksize2 + 1 + k;
          sy = sy >= 0 ? (sy < sH ? sy : sH - 1) : 0;

          for (k1 = std::max(k1, k); k1 < ksize; k1++)
          {
              if (k1 > k && sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
              {// k1 = 1, k = 0
                  memcpy(rows[k], rows[k1], dW * sizeof(rows[0][0]));
                  //memcpy(rows[0], rows[1], dW * sizeof(int0));
              }
          }
          if (k1 == ksize)
              k0 = std::min(k0, k); // remember the first row that needs to be computed
          srows[k] = src + sy*sW*cn;
          prev_sy[k] = sy;
      }

      const unsigned char* S0 = srows[0], * S1 = srows[1];
      int* D0 = rows[0], * D1 = rows[1];
      if (k0 < ksize) {
          //hresize((const T**)(srows + k0), (WT**)(rows + k0), ksize - k0, xofs, (const AT*)(alpha), ssize.width, dsize.width, cn, xmin, xmax);
          for (dx = 0; dx < xmax; dx++) {
              D0[dx] = S0[xofs[dx]] * alpha[dx * 2] + S0[xofs[dx] + 1] * alpha[dx * 2 + 1];
              D1[dx] = S1[xofs[dx]] * alpha[dx * 2] + S1[xofs[dx] + 1] * alpha[dx * 2 + 1];
          }
          for (; dx < width; dx++) {
              D0[dx] = S0[xofs[dx]] * INTER_RESIZE_COEF_SCALE;
              D1[dx] = S1[xofs[dx]] * INTER_RESIZE_COEF_SCALE;
          }
      }
      // vresize( (const WT**)rows, (T*)(dst.data + dst.step*dy), beta, dsize.width );
      for (dx = 0; dx < width; dx++) {
          // VResizeLinear:1926
          dst[dy * width + dx] = static_cast<unsigned char>(( ((beta[dy * 2] * (D0[dx]>>4))>>16) + ((beta[dy * 2 + 1] * (D1[dx]>>4))>>16) +2)>>2) ;
      }
  }

  cv::Mat resized_img(dH, dW, CV_8UC1);
  memcpy(resized_img.data, dst, sizeof(unsigned char)*dW*dH);
#endif
  cv::imshow("resized ", resized_img);
  cv::waitKey();
  return 0;
}
