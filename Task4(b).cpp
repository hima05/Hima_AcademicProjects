/*
		HPC Lab project
			
		 Areti Hima Sai Kiran	(216100287)
								

Task 4b : Generating a large 1000x1000 matrix with random numbers and displaying Heat Map of corresponding Matrix (Parallel programming)
*/
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<iostream>
#include <vector>
#include<omp.h>
using namespace std;
using namespace cv;

void heatMap(Mat img,double max,double min,double mid,double M_A[1000][1000]); //funtion declaration

int main() {


	double max, min, mid; // Variables for different colour ranges



	static double Mat_A[1000][1000]; // declaration of 1000x1000 matrix
	srand(1234);

	//parallel programming
	/* Generating matrix elements with random numbers*/
#pragma omp parallel for
	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < 1000; j++) {
			Mat_A[i][j] = rand() ;

		}

	}

	/*Calculating max and min values of the matrix elements*/
	max = Mat_A[0][0];
	min = Mat_A[0][0];

	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < 1000; j++) {

			if (Mat_A[i][j] > max) {
				max = Mat_A[i][j];
			}
			if (Mat_A[i][j] < min) {
				min = Mat_A[i][j];
			}
		}

	}


	mid = (max + min) / 2; // Average value of Matric elements value

	
	char window_heatmap[] = "Heatmap of Matrix A 1000x1000";  // Title of the image window
	

	
	Mat hMap = Mat::zeros(1920, 1500, CV_8UC3);    // creates image


	resizeWindow(window_heatmap, 1920, 1500); // resizes the window

	heatMap(hMap,max,min,mid,Mat_A);  //Funtion calling with required parameters


	
	imshow(window_heatmap, hMap);   //Display image window
	vector<int> compression_params;  //Vector for PNG compression parametrers
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); 
	compression_params.push_back(0);  // 0 for min compression and max quality
	imwrite("Mat.png", hMap, compression_params);  // Create image with file with the given title
	waitKey(0);    
	return(0);






}

/*Computing image data*/
void heatMap(Mat img,double max,double min,double mid, double M_A[1000][1000])
{
	int x;
	int lineType = LINE_8;
	double mid1, mid2;
	mid1 = (max + mid) / 2; 
	mid2 = (mid + min) / 2;    //Mid values for RGB distribution

//parallel
	/* Creates each pixel according to the each matrix element */
#pragma omp parallel for
	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < 1000; j++) {


			
			Point pixel[1][4];

			pixel[0][0] = Point((j+100), i);
			pixel[0][1] = Point((j + 1) + 100, i);
			pixel[0][2] = Point((j + 1) + 100, (i + 1) );
			pixel[0][3] = Point((j + 100), (i + 1) );

			x = (j + 1) + 100;
		

			const Point* ppt[1] = { pixel[0] };
			int npt[] = { 4 };
			/* Assigning colour to each pixel according to the element value */

			/* Red to yellow */
			if (M_A[i][j] > mid1) {

				fillPoly(img,
					ppt,
					npt,
					1,
					Scalar(0, 255 *(1- ((M_A[i][j] - mid1) / (max - mid1))), 255),
					lineType);


			}

			/* Yellow to Green */
			else if (M_A[i][j] <= mid1  && M_A[i][j] > mid) {
				fillPoly(img,
					ppt,
					npt,
					1,
					Scalar(0, 255 , 255*(((M_A[i][j] - mid) / (mid1 - mid)))),
					lineType);
			}




			/* Green to Mild blue */
			else if (M_A[i][j] < mid  && M_A[i][j] > mid2) {
			fillPoly(img,
				ppt,
				npt,
				1,
				Scalar(255 *(1- (((mid-M_A[i][j]) / (mid - mid2)))),255 , 0),
				lineType);
		}
			/* Mild blue to Blue */
			else if (M_A[i][j] <= mid2) {
				fillPoly(img,
					ppt,
					npt,
					1,
					Scalar(255 , 255 * (1 - (((mid2 - M_A[i][j]) / (mid2 - min)))),0 ),
					lineType);
				
			}
			/*Green*/
			else {
				fillPoly(img,
					ppt,
					npt,
					1,
					Scalar(0, 255, 0),
					lineType);
			}
			


			


		}

	}



//parallel
	 /* Scale to describe the meaning of each colour */
#pragma omp parallel for
	for (int i = 0; i < 600; i++) {
		for (int j = 0; j < 50; j++) {



			Point pixel1[1][4];

			pixel1[0][0] = Point((x+j + 100), i+100);
			pixel1[0][1] = Point((x + j + 1) + 100, i+100);
			pixel1[0][2] = Point((x + j + 1) + 100, (i + 1)+100);
			pixel1[0][3] = Point((x + j + 100), (i + 1)+100);

			


			const Point* ppt1[1] = { pixel1[0] };
			int npt1[] = { 4 };

			//Red to yellow
			if (i <= 150) {

				fillPoly(img,
					ppt1,
					npt1,
					1,
					Scalar(0, 255*((double)i/150), 255),
					lineType);
			}
			//yellow to Green
			if (i > 150 && i <= 300) {
				fillPoly(img,
					ppt1,
					npt1,
					1,
					Scalar(0, 255 , 255*((double)( 1-(double)(i-150)/150))),
					lineType);
			}
			//Green to Mild blue
			if (i > 300 && i <= 450) {
				fillPoly(img,
					ppt1,
					npt1,
					1,
					Scalar(255*((double) (i-300)/150), 255,0),
					lineType);
			}
			//Mild blue to Blue
			if (i > 450) {
				fillPoly(img,
					ppt1,
					npt1,
					1,
					Scalar(255, 255 * ((double)(1 - (double)(i - 450) / 150)), 0),
					lineType);
			}


			
			
		}
	}

	//Scale description

	putText(img, "MAX", Point(x+100, 90),
		FONT_HERSHEY_COMPLEX, 0.9, Scalar(255, 255, 255), 1, 8);

	putText(img, "MIN", Point(x + 100, 740),
		FONT_HERSHEY_COMPLEX, 0.9, Scalar(255, 255, 255), 1, 8);
}

