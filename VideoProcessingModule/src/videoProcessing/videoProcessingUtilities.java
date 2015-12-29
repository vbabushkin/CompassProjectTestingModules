package videoProcessing;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;

public class videoProcessingUtilities {
	static String RGBfilename;
	static String Depthfilename;
	
	static String  currentFolder = "GRAY_LIGHT_1_20";
    static int kinectID=1;

	static String videoBckgDirectory =".//videoFrames//bckgVideo_"+kinectID+"//";
	static String depthBckgDirectory =".//videoFrames//bckgDepth_"+kinectID+"//";
	
	static File bckgImageFile = new File(videoBckgDirectory+"frame_"+"outVideo_"+203+".jpg");
	static File bckgDepthFile =  new File(depthBckgDirectory+"frame_"+"outDepth_"+203+".jpg");
    
    public static void main(String args[]){
    	//TODO:
    	//extract premasked image without background
    	//overlay initial mask
    	//calculate average pixel values in the overlayed region
    	//based on this number make decision which color range to use
    	
    	//load native library
    	System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
    	int index =173;//182;
	    
    	//read serialized images
    	 String framesPath ="./videoFrames/"+ currentFolder+"//";
		String depthDirectory = framesPath+"outDepth//";
	
		String videoDirectory = framesPath+"outVideo//";
		
	
		File videoImageFile = new File(videoDirectory+"frame_"+"outVideo_"+index);
		File depthImageFile = new File(depthDirectory+"frame_"+"outDepth_"+index);
		File bckgImageFile = new File(videoBckgDirectory+"frame_"+"outVideo_"+203);
		File bckgDepthFile =  new File(depthBckgDirectory+"frame_"+"outDepth_"+203);
		
		
		Mat depthBackgFrame = reconstructImage(bckgDepthFile,2);
//		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(depthBackgFrame), "depthBackground serialized");
		
		Mat rgbBackgFrame = reconstructImage(bckgImageFile,1);
//		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(rgbBackgFrame), "background serialized");
	    
		Mat depthFrame = reconstructImage(depthImageFile,2);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(depthFrame), "depthFrame serialized");
		
		
		MatOfInt parameters = new MatOfInt();
		parameters.fromArray(Highgui.CV_IMWRITE_JPEG_QUALITY, 100);
		//Highgui.imwrite("./images/depthFrame_"+index+".jpg",depthFrame);
		
		Mat rgbFrame = reconstructImage(videoImageFile,1);
    	ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(rgbFrame, new Size(960,540))), "colorFrame serialized");
    			
	
	    //remove background before homography
	    
	    Mat depthFrameBckgSubtracted = subtractBackground(depthFrame, depthBackgFrame);
	    
	    //then perform a homography:
	    
	    Mat hDepthFrameBckgSubtracted = performHomographyTransformation(depthFrameBckgSubtracted, new Size(1920,1080));
	    
	    
//	    Imgproc.erode(depthFrameBckgSubtracted, depthFrameBckgSubtracted, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5)));
//	  	Imgproc.dilate(depthFrameBckgSubtracted, depthFrameBckgSubtracted, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5)));
//	    
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(hDepthFrameBckgSubtracted, new Size(960,540))),"Segmented Depth Frame # "+index);

	    //to fill the gaps
	    Imgproc.threshold(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted,0, 255, Imgproc.THRESH_BINARY);
	    
	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    //remove holes
	    //Mat copyOfhDepthFrameBckgSubtracted =removeHolesFromContour( hDepthFrameBckgSubtracted.clone());
	  
	    
	    MatOfPoint largestContour = extractLargestContour(hDepthFrameBckgSubtracted.clone(),25000, 120000);
	    
	    //calculate convex hull
	    
	    ArrayList<MatOfPoint> hullContours = calculateConvexHull(largestContour);
	    
	    Mat mask = getContourMasked(hDepthFrameBckgSubtracted.clone(), hullContours.get(0));
	    
	    //Mat mask = getContourMasked(hDepthFrameBckgSubtracted.clone(), largestContour);
	    
	    //draw the largest contour
	    drawCurrentContour(hDepthFrameBckgSubtracted.clone(),   largestContour,"Segmented ");
	    
	    //draw the contour mask
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(mask, new Size(960,540))),"mask # "+index);
		  
	    //get segmented RGB image
	    //display
	    Mat segmentedRGBFrame = segmentRGBImage(rgbFrame,rgbBackgFrame);
	    
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(segmentedRGBFrame, new Size(960,540))),"segmented RGB Frame");
        
        //then mask it with mask from depthFrame 
        Mat preMaskedRGBFrame= new Mat();
        segmentedRGBFrame.copyTo(preMaskedRGBFrame, mask); 
        Imgproc.equalizeHist(preMaskedRGBFrame, preMaskedRGBFrame);
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(preMaskedRGBFrame, new Size(960,540))),"pre masked RGB Frame");
 
	    //extract rectangle with masked shoe only
        //find bounding minarea rectangle corresponding to largest contour
        MatOfPoint2f pointsPreROI = new MatOfPoint2f(largestContour.toArray() ); 
        RotatedRect preROI = Imgproc.minAreaRect(pointsPreROI);
	    
        Mat preMaskedRGBFrameROI = rotateExtractedShoeprint(preMaskedRGBFrame, preROI, new Size(500,750), 2);
        //get image containing the masked shoe only
        //Imgproc.getRectSubPix(preMaskedRGBFrame, preROI.size, preROI.center, preMaskedRGBFrameROI);
        
        
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(preMaskedRGBFrameROI),"preMaskedRGBFrameROI");
//        
//        int lowThreshold=100;
//	    int ratio = 250;
//	    int kernel_size = 3;
//	    Mat cropped = preMaskedRGBFrameROI.clone();
//	    
//	    Imgproc.adaptiveThreshold(cropped, cropped, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
//	    Imgproc.equalizeHist(cropped, cropped); 
//	    //Imgproc.threshold(cropped, cropped, 0, 255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
//	    //
//	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(cropped),"THRESHOLDED");
//        //
//	    Mat dstPreMaskedCanny=new Mat();
//	    Imgproc.Canny(cropped, dstPreMaskedCanny, lowThreshold, ratio, kernel_size,true);
//	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(dstPreMaskedCanny),"preMaskedImage Canny");
////        
        
        
        
        //get the pixels
//        double pixelCount = 0.0;
//        for(int row =0;row < preMaskedRGBFrameROI.rows(); row++){
//        	for(int col = 0; col < preMaskedRGBFrameROI.cols(); col++){
//        		double[] data = preMaskedRGBFrameROI.get(row, col);
//        		pixelCount+=data[0];
//        		
//        	}
//        	
//        }
        
      //get the pixels
        double pixelCount = 0.0;
        for(int row =0;row < preMaskedRGBFrame.rows(); row++){
        	for(int col = 0; col < preMaskedRGBFrame.cols(); col++){
        		double[] data = preMaskedRGBFrame.get(row, col);
        		pixelCount+=data[0];
        		
        	}
        	
        }
        
        
	    
	    
	    
	    
	    
	    
	    
	    ////may be first remove noise and then segment?
	    //Imgproc.blur(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted, new Size(3,3));
		//Imgproc.threshold(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted, 0, 255, Imgproc.THRESH_BINARY);
	 
	    //overlay RGB and Depth images to check
	    Imgproc.cvtColor(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted, Imgproc.COLOR_GRAY2BGR);
	    Mat overlayed = overlayImages(rgbFrame,hDepthFrameBckgSubtracted);

	    //segment overlayed image
	    Mat overlayedSegmented = new Mat();
	    Core.inRange(overlayed, new Scalar(215, 215, 215), new Scalar(230, 230, 230), overlayedSegmented);//215 and 225 for 1_20
	    
	    //Core.inRange(overlayed, new Scalar(220, 220, 220), new Scalar(225, 225, 225), overlayedSegmented);
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(overlayedSegmented, new Size(960,540))),"Segmented OVERLAYED Frame # "+index);
	    
	    //Highgui.imwrite("./images/Increase.jpg",overlayedSegmented);
	    
	    ///////////////////////////////////////////////////////////////////////////////////////////////////////
	    //now find contours, hulls and extract shoes
	    //check if shoelike object is detected and return detected object's contours as ArrayList
	   if( shoeLikeObjectDetected(overlayedSegmented.clone(), 25000, 120000)){
		   System.out.println("DETECTED ");
	   }
	    
	   
	   //Imgproc.cvtColor(overlayedSegmented, overlayedSegmented, Imgproc.COLOR_BGR2GRAY);
	   
	   Mat copyOfhDepthFrameBckgSubtracted =removeHolesFromContour( overlayedSegmented.clone());
	   
	   ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(copyOfhDepthFrameBckgSubtracted, new Size(960,540))),"gaps removed "+index);
	   ///////////////////////////////////////////////////////////////////////////////////////////////////////
	   //get convex hull contours for final mask
	   //to store candidate contours
	   ArrayList<MatOfPoint> resContours = new ArrayList<MatOfPoint>();
	   List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	   
	   //recognize contours
	   Imgproc.findContours(copyOfhDepthFrameBckgSubtracted.clone(), contours, new Mat(), Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
		 
	 //TODO: ADD logic for selecting of proper contour
	 		for(MatOfPoint contour: contours){
	 		 	MatOfPoint2f approx =  new MatOfPoint2f();
	 		 	MatOfPoint2f newMat = new MatOfPoint2f(contour.toArray() ); 
	 	        int contourSize = (int)contour.total();
	 	        //approximate with polygon contourSize*0.05 -- 5% of curve length, curve is closed -- true
	 	   	 	Imgproc.approxPolyDP(newMat, approx, contourSize*0.07, true);//0.035
	 	   	 	
	 	   	 	MatOfPoint approxMatOfPoint = new MatOfPoint(approx.toArray());
	 	   	 	
	 	   	if( Imgproc.isContourConvex(approxMatOfPoint) && Math.abs(Imgproc.contourArea(approxMatOfPoint))>25000 && Math.abs(Imgproc.contourArea(approxMatOfPoint))<120000 ){
	 	   		
	 	   		resContours.add(approxMatOfPoint);
	 	   		drawCurrentContour(copyOfhDepthFrameBckgSubtracted.clone(), approxMatOfPoint,"best contour");
	 	   		}
	 		}
	   
	 		System.out.println("SIZE OF RESULTING CONTOURS: "+resContours.size());
	 		
	   
		//recognize largest contours
	   MatOfPoint finalMaxContour = extractLargestContour(copyOfhDepthFrameBckgSubtracted.clone(),25000, 120000);
	    
	    //calculate convex hull
	    
	    ArrayList<MatOfPoint> finalHullContours = calculateConvexHull(finalMaxContour);
	    Imgproc.cvtColor(copyOfhDepthFrameBckgSubtracted, copyOfhDepthFrameBckgSubtracted, Imgproc.COLOR_GRAY2BGR);
	    Imgproc.drawContours(copyOfhDepthFrameBckgSubtracted, finalHullContours, -1, new Scalar(0,255,0), 2);
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(copyOfhDepthFrameBckgSubtracted, new Size(960,540))),"gaps removed "+index);
		   
	    //drawHullContours(copyOfhDepthFrameBckgSubtracted, finalMaxContour);
	   ///////////////////////////////////////////////////////////////////////////////////////////////////////
	    MatOfPoint2f finalMaskedPointsROI = new MatOfPoint2f(finalHullContours.get(0).toArray() ); 
        RotatedRect finalROI = Imgproc.minAreaRect(finalMaskedPointsROI);
        //Mat finalMaskedRGBFrameROI = rotateExtractedShoeprint(preMaskedRGBFrame, finalROI, new Size(500,750), 2);
        //get image containing the masked shoe only
        //Imgproc.getRectSubPix(preMaskedRGBFrame, preROI.size, preROI.center, preMaskedRGBFrameROI);
        
        
        //ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(finalMaskedRGBFrameROI),"FINALLY");
        
	    Mat newMask = getContourMasked(copyOfhDepthFrameBckgSubtracted,finalHullContours.get(0));
	    
	    
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(newMask, new Size(960,540))),"mask");
//	    //mask registred  
        Mat imageROIRegistred= new Mat();
//        //Imgproc.cvtColor(preMaskedRGBFrame, preMaskedRGBFrame, Imgproc.COLOR_BGR2GRAY);
        preMaskedRGBFrame.copyTo(imageROIRegistred, newMask); 
//	    
        Mat MaskedRGBFrameROI = rotateExtractedShoeprint(imageROIRegistred, finalROI, new Size(500,750), 2);
//           
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(MaskedRGBFrameROI),"FINALLY");
//	    
        Imgproc.adaptiveThreshold(MaskedRGBFrameROI, MaskedRGBFrameROI, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(MaskedRGBFrameROI),"ADAPTIVE THRESHOLD");
	    
	   System.out.println("Done..........");
	    
	   System.out.println("pixelCount "+pixelCount);
	   System.out.println("AVERAGE PIXEL VALUE "+ pixelCount/(preMaskedRGBFrameROI.width()*preMaskedRGBFrameROI.height()));
	   System.out.println("Shoe Area "+ Imgproc.contourArea(finalMaxContour));
	    
	    
    }//END OF MAIN
    
    /**
     * Additional morphological operations
     */
    
    public static void purifyContour(Mat image){
    	ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(image),"original");
		
		int alpha = 2;
		int beta = 50;
		image.convertTo(image, -1 , alpha, beta);	
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(image),"increased brightness");
		
		
		Mat threshCropped = new Mat();
		
		//Imgproc.threshold(cropped, threshCropped, 50, 255, Imgproc.THRESH_BINARY);
		
		Imgproc.equalizeHist(image, threshCropped);
		Imgproc.threshold(threshCropped, threshCropped, 240, 255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
		//Imgproc.adaptiveThreshold(cropped, threshCropped, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, 5);
//	    
		
		
		//initial contour without dilation
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(threshCropped),"OTSU 1");
		
		
		
//		Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(24, 24));
//		Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));
//		
//		
//		Imgproc.dilate(cropped, cropped, dilateElement);
//		Imgproc.erode(cropped, cropped, erodeElement);
		
		
		//Imgproc.erode(cropped, cropped, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15,15)));
		Imgproc.dilate(image, image, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15,15)));
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(image),"dilated");
		
		Imgproc.equalizeHist(image, image);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(image), "equilized");
		
		
		Mat testCropped = image.clone();
		org.opencv.core.Core.inRange(testCropped, new Scalar(130, 130, 130), new Scalar(255, 255, 255), testCropped);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(testCropped), "testCropped");
		
		
		
		Imgproc.threshold(image, image, 120, 255, Imgproc.THRESH_BINARY);
		
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(image),"extracted 1");
		
		//add border to cropped
		double  borderConstant =0.1;
		int top, bottom, left, right;
		/// Initialize arguments for the filter
        top = (int) (borderConstant*image.rows()); 
        bottom = (int) (borderConstant*image.rows());
        left = (int) (borderConstant*image.cols()); 
        right = (int) (borderConstant*image.cols());
        Mat src=new Mat();
        //destination = source;
        Imgproc.copyMakeBorder(image, src, top, bottom, left, right, Imgproc.BORDER_CONSTANT, new Scalar(255,255,255));
	
		image = src;
		


		
		
		
		//Imgproc.GaussianBlur(cropped, cropped, new Size(9, 9), 0);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(image),"extracted ");
        
    }
    
    
    /**
     * 
     * @param image
     * @param background
     * @return
     */
     public static Mat segmentRGBImage(Mat image, Mat background){
    	 //pre-segment the colored image
	        //first subtract a background
	        Mat backgroundSubtracted = new Mat();
//	        if(image.channels()<3){
//	        	Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2BGR);
//				Imgproc.cvtColor(background, background, Imgproc.COLOR_GRAY2BGR);
//				
//	        }
	    		
			//Core.subtract(currentImageFrame, videoBackground, backgroundSubtracted);
			
			
			//applying background remover
//			 BackgroundSubtractorMOG2 bg=new BackgroundSubtractorMOG2();	
//			 double learningRate = 0.01;
//			 bg.apply(background, image.clone(),learningRate);
//			 bg.apply(image.clone(), backgroundSubtracted, learningRate);
			 Core.subtract(background, image.clone(), backgroundSubtracted);

			
			ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(backgroundSubtracted, new Size(960,540))),"backgroundSubtracted");
	        

	        Mat preMask = new Mat();
			Imgproc.GaussianBlur(backgroundSubtracted, backgroundSubtracted, new Size(5, 5), 0);
			Imgproc.threshold(backgroundSubtracted, preMask, 40, 255, Imgproc.THRESH_BINARY);
			
			//Imgproc.cvtColor(preMask, preMask, Imgproc.COLOR_BGR2GRAY);
//			Imgproc.erode(preMask, preMask, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));
//	      Imgproc.erode(preMask, preMask, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));
	        Imgproc.GaussianBlur(preMask, preMask, new Size(3,3), 0);
			Imgproc.threshold(preMask, preMask, 0, 255, Imgproc.THRESH_OTSU);
			
			
			Mat preMaskedImage= new Mat();
			image.copyTo(preMaskedImage, preMask); 
	        
//			//mask the shoe contour from pre-masked image
//			Mat finallyMasked = new Mat();
//			preMaskedImage.copyTo(finallyMasked, colorMask);
//			
//			//equalize histograms to make image features more pronounced
//			//Imgproc.cvtColor(finallyMasked, finallyMasked, Imgproc.COLOR_BGR2GRAY);
//	        Imgproc.equalizeHist(finallyMasked, finallyMasked);
	        
	       return preMaskedImage;
     }
    
     /**
 	 * 
 	 * @param image
 	 * @return
 	 */
 	public static List<MatOfPoint> extractAllDetectedContours(Mat image, int MIN_AREA, int MAX_AREA){
 		  //to store the recognized contours
 				List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
 				//recognize contours
 				Imgproc.findContours(image.clone(), contours, new Mat(), Imgproc.RETR_LIST,Imgproc.RETR_LIST);
 				List<MatOfPoint> largestContours  = new ArrayList<MatOfPoint>();
 				
 				//TODO: ADD logic for selecting of proper contour
 				for(MatOfPoint contour: contours){
 					MatOfPoint2f approx =  new MatOfPoint2f();
 		 		 	MatOfPoint2f newMat = new MatOfPoint2f(contour.toArray() ); 
 		 	        int contourSize = (int)contour.total();
 					Imgproc.approxPolyDP(newMat, approx, contourSize*0.0015, true);//0.0035
 			
 		 	   	 	if(Math.abs(Imgproc.contourArea(contour))>MIN_AREA && Math.abs(Imgproc.contourArea(contour))<MAX_AREA){   	 		
 		 	   	 		largestContours.add(contour);
 		 	   	 	}//end of if
 		 	   	 	
// 		 	   	 	MatOfPoint approxMatOfPoint = new MatOfPoint(approx.toArray());
// 					if(Math.abs(Imgproc.contourArea(approxMatOfPoint))>MIN_AREA && Math.abs(Imgproc.contourArea(approxMatOfPoint))<MAX_AREA && Math.abs(Imgproc.contourArea(approxMatOfPoint))> Math.abs(Imgproc.contourArea(largestContour))){
// 						largestContour=approxMatOfPoint;
 
 				}//end of for
 				return largestContours;
 	}
     
    /**
	 * 
	 * @param image
	 * @return
	 */
	public static MatOfPoint extractLargestContour(Mat image, int MIN_AREA, int MAX_AREA){
		//to store the recognized contours
		 List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		//recognize contours
		 Imgproc.findContours(image, contours, new Mat(), Imgproc.RETR_LIST,Imgproc.RETR_LIST);
		 MatOfPoint largestContour = contours.get(0);
			//TODO: ADD logic for selecting of proper contour
			for(MatOfPoint contour: contours){
				MatOfPoint2f approx =  new MatOfPoint2f();
	 		 	MatOfPoint2f newMat = new MatOfPoint2f(contour.toArray() ); 
	 	        int contourSize = (int)contour.total();
				Imgproc.approxPolyDP(newMat, approx, contourSize*0.0015, true);//0.0035
	 	   	 	
	 	   	 	MatOfPoint approxMatOfPoint = new MatOfPoint(approx.toArray());
	 	   	 	
	 	   	 	if(Math.abs(Imgproc.contourArea(contour))>MIN_AREA && Math.abs(Imgproc.contourArea(contour))<MAX_AREA && Math.abs(Imgproc.contourArea(contour))> Math.abs(Imgproc.contourArea(largestContour))){
	 	   	 		largestContour=contour;
	 	   	 	
//				if(Math.abs(Imgproc.contourArea(approxMatOfPoint))>MIN_AREA && Math.abs(Imgproc.contourArea(approxMatOfPoint))<MAX_AREA && Math.abs(Imgproc.contourArea(approxMatOfPoint))> Math.abs(Imgproc.contourArea(largestContour))){
//					largestContour=approxMatOfPoint;
				}
					
			}
			return largestContour;
	}//end of extractShoeContour
    
    
	/**
	 * 
	 * @param image
	 * @param tempContour
	 * @return
	 */
	public static Mat getContourMasked(Mat image, MatOfPoint tempContour){
		// Create a mask for each contour to mask out that region from image.
        Mat mask = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
        mask.setTo(new Scalar(0,0,0)) ;
        List <MatOfPoint> largestDetectedContours = new ArrayList<MatOfPoint>();
        largestDetectedContours.add(tempContour);
        //place the white filled contour on the mask
        Imgproc.drawContours(mask,  largestDetectedContours, -1, new Scalar(255,0,0),-1); 
        
        //erode and dilate the mask to remove sharp edges
        Imgproc.dilate(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));
        Imgproc.erode(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));
		Imgproc.dilate(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7,7)));
		Imgproc.blur(mask, mask, new Size(5,5));
		Imgproc.threshold(mask, mask, 0, 255, Imgproc.THRESH_OTSU);
        return mask;
	}
	
	/**
	 * 
	 * @param contour
	 * @return
	 */
	public static ArrayList<MatOfPoint> calculateConvexHull(MatOfPoint contour){
		MatOfInt hull = new MatOfInt();
		Imgproc.convexHull(contour, hull, false);//tempMaskContour
		
		MatOfPoint mopOut = new MatOfPoint();
		mopOut.create((int)hull.size().height,1,CvType.CV_32SC2);
		
		for(int i = 0; i < hull.size().height ; i++){
		
			int idx = (int)hull.get(i, 0)[0];
			double[] point = new double[] {
					contour.get(idx, 0)[0], contour.get(idx, 0)[1]
			};
			mopOut.put(i, 0, point);
		} 
		ArrayList<MatOfPoint> hullContours = new ArrayList<MatOfPoint>();
		hullContours.add(mopOut);
		return hullContours;
	}
	
	/**
	 * 
	 * @param image
	 * @param contour
	 */
	public static Mat drawHullContours(Mat image, MatOfPoint contour){
		if(image.channels()<3)
    		Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2BGR);
		MatOfInt hull = new MatOfInt();
		Imgproc.convexHull(contour, hull, false);//tempMaskContour
		
		MatOfPoint mopOut = new MatOfPoint();
		mopOut.create((int)hull.size().height,1,CvType.CV_32SC2);
		
		for(int i = 0; i < hull.size().height ; i++){
		
			int idx = (int)hull.get(i, 0)[0];
			double[] point = new double[] {
					contour.get(idx, 0)[0], contour.get(idx, 0)[1]
			};
			mopOut.put(i, 0, point);
		} 
		ArrayList<MatOfPoint> hullContours = new ArrayList<MatOfPoint>();
		hullContours.add(mopOut);
		
		Imgproc.drawContours(image, hullContours, -1, new Scalar(0,255,0), 2);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(image, new Size(960,540))),"HULL CONTOURS");

		return image;
	}
	
    /**
     * 
     * @param image
     * @return
     */
    public static Mat removeHolesFromContour(Mat image){
    	 
	    //http://stackoverflow.com/questions/15806968/edge-smoothing-and-filling-inner-contours-in-opencv-with-ios
    	
    	Mat imageCopy = image.clone();
    	
	    Imgproc.morphologyEx(imageCopy,imageCopy,Imgproc.MORPH_GRADIENT,Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(3,3)));
	    
	    Core.bitwise_not(imageCopy,imageCopy);
	    //ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(imageCopy, new Size(960,540)))," bitwise ");
	    
	    
	    Mat smallholes=Mat.zeros(imageCopy.size(), CvType.CV_8UC1);
	    ArrayList<MatOfPoint> contours= new ArrayList<MatOfPoint>(); 
	    Imgproc.findContours(imageCopy.clone(), contours, new Mat(), Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
	    for(int i = 0; i < contours.size(); i++)
	    {       
	        double area = Imgproc.contourArea(contours.get(i));
	        if(area<30000)
	        	Imgproc.drawContours(smallholes, contours, i, new Scalar(255,255,255), -1);
	    }
	    
	    Mat result = new Mat();
	    Core.bitwise_or(image,smallholes,result);
	   // Imgproc.morphologyEx(result,result,Imgproc.MORPH_CLOSE,Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(3,3)));
	    
	    
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(result, new Size(960,540))),"Gaps filled ");
	    return result;
	    
    }
    
    
    /**
     * 
     * @param image
     * @param contours
     * @param currentContour
     * @return
     */
    public static void drawCurrentContour(Mat image,  MatOfPoint currentContour, String label){
    	if(image.channels()<3)
    		Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2BGR);
    	ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    	contours.add(currentContour);
    	Imgproc.drawContours(image, contours, -1, new Scalar(0,0,255),3);
    	ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizeImage(image, new Size(960,540))),label);
 	    
    }
    
    
    
    /**
	 * 
	 * @param segmentedDepthFrame
	 * @return
	 */
	public static boolean shoeLikeObjectDetected(Mat image, int MIN_AREA, int MAX_AREA){
		
		Mat dst = new Mat();

		Imgproc.threshold(image, dst, 0, 255, Imgproc.THRESH_BINARY);
		
		int count =0;
		
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
		for(MatOfPoint contour:contours){
			System.out.println("Contours area "+Imgproc.contourArea(contour));
			
			if( Math.abs(Imgproc.contourArea(contour))>MIN_AREA && Math.abs(Imgproc.contourArea(contour))<MAX_AREA ){
				
				drawCurrentContour(image, contour,"Segmented Contour");
				count++;
			}
		}
 
		//System.out.println("Size of candidate contours "+candidateContours.size());
		
		if(count!=0){
			System.out.println(count + "	shoes are detected");;
			return true;
		}
		else {
			System.out.println("nothing is detected");
			return false;
		}
			
	}
    
    
    
    
    
    /**
     * 
     * @param depthFrame
     * @param depthBackgFrame
     * @return
     */
    public static Mat subtractBackground(Mat depthFrame, Mat depthBackgFrame){
    	//first subtract a background
	    //1. convert everything to grayscale
    	if(depthFrame.channels()>1)
    		Imgproc.cvtColor(depthFrame, depthFrame, Imgproc.COLOR_BGR2GRAY);
    	if(depthBackgFrame.channels()>1)
    		Imgproc.cvtColor(depthBackgFrame, depthBackgFrame, Imgproc.COLOR_BGR2GRAY);
    	
	    //2. segment depth frame
	    Mat segmentedDepthFrame = new Mat();
	    Core.inRange(depthFrame, new Scalar(54, 54, 54), new Scalar(55, 55, 55), segmentedDepthFrame);
	    
	    //3. segment depth background frame
	    Mat segmentedDepthBckgFrame = new Mat();
	    Core.inRange(depthBackgFrame, new Scalar(54, 54, 54), new Scalar(55, 55, 55), segmentedDepthBckgFrame);
	    
	    
	    //4. ordinary subtract
        Mat backgroundSubtracted = new Mat();
        Core.subtract(segmentedDepthFrame,segmentedDepthBckgFrame, backgroundSubtracted);
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(backgroundSubtracted),"background Subtracted");
        return backgroundSubtracted;
    }
  
	
	
    
    /**
	 * 
	 * @param image
	 * @param mask
	 */
	public static Mat overlayImages(Mat image, Mat mask){
		
		//1. convert everything to grayscale
    	if(image.channels()<3)
    		Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2BGR);
    	if(mask.channels()<3)
    		Imgproc.cvtColor(mask, mask, Imgproc.COLOR_GRAY2BGR);
		
		double alpha=0.16;
        double beta = 1.0 -alpha;
        double gamma = 0.0;
        Mat result = new Mat();
        Core.addWeighted(image, alpha, mask, beta, gamma, result);
        Mat resizedImage = resizeImage(result, new Size(960,540));
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(resizedImage), "Mask for RGB data");
        MatOfInt parameters = new MatOfInt();
        parameters.fromArray(Highgui.CV_IMWRITE_JPEG_QUALITY, 99);
        Highgui.imwrite("./images/maskedRGB.jpg",result);
        return result;
	}
	
	/**
	 * 
	 * @param image
	 * @param size
	 * @return
	 */
	public static Mat resizeImage(Mat image, Size size){
		Mat resizedImage = new Mat();
        Imgproc.resize( image, resizedImage, size );
        return resizedImage;
	}
    
    /**
	 * 
	 * @param image
	 * @param size
	 * @return
	 */
	public static Mat performHomographyTransformation(Mat image, Size size){
		Mat H = new Mat(3, 3, CvType.CV_64F);
        H.put(0, 0, 2.884206871130816);
        H.put(0, 1, -0.01595790917398606);
        H.put(0, 2, 325.2537036078101);
        H.put(1, 0, -0.02891720553069146);
        H.put(1, 1, 2.885108295367276);
        H.put(1, 2, -55.91800388490574);
        H.put(2, 0, 3.555978795624555e-006);
        H.put(2, 1, -1.95047104825213e-005);
        H.put(2, 2, 1.0);
        
        Mat hImage=new Mat();
		Imgproc.warpPerspective(image, hImage, H, size);
		return hImage;
	}//end of performHomographyTransformation
	
	
	
	/**
	 * 
	 * @param imgFile
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static Mat reconstructImage(File imgFile, int index) {
		 Mat mat = new Mat();
		
        FileInputStream fis;
		try {
			fis = new FileInputStream(imgFile.getPath());
			ObjectInputStream ois = new ObjectInputStream(fis);
	        byte [] data=(byte[]) ois.readObject();

		   
		    
		    if(index == 1){
		    	mat = new Mat(1080,1920, CvType.CV_8UC3);
		    }
		    if(index == 2){
		    	mat = new Mat(424,512, CvType.CV_8UC3);
		    }
		    
		    
	        mat.put(0, 0, data);
	        
		    
	        ois.close();
	        fis.close();
	
	        
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return mat;
		
	}
	
	/**
	 * rotate extracted shoeprint to make it vertical, place it in center and crop it to desired size 
	 * @param image
	 * @param rect
	 * @param borderConstant
	 * @return
	 */
	static Mat rotateExtractedShoeprint(Mat image, RotatedRect rect, Size size, double borderConstant){
		
		Mat result=new Mat();
		Mat rotated=new Mat();
		
		int top, bottom, left, right;
		/// Initialize arguments for the filter
        top = (int) (borderConstant*image.rows()); 
        bottom = (int) (borderConstant*image.rows());
        left = (int) (borderConstant*image.cols()); 
        right = (int) (borderConstant*image.cols());
        Mat src=new Mat();
        //destination = source;
        Imgproc.copyMakeBorder(image, src, top, bottom, left, right, Imgproc.BORDER_CONSTANT, new Scalar(0,0,0));
	
        
		//for(int i=0;i<masks.size();i++){
		//RotatedRect rect= masks;//.get(i);

		// matrices we'll use
		
		// get angle and size from the bounding box
		float angle = (float) rect.angle;
		Size rect_size = rect.size;
		// thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
		if (rect.angle < -45.) {
		    angle += 90.0;
		    //swap width and height
		    rect_size=new Size(rect_size.height, rect_size.width);
		}
		
		rect.center=new Point(rect.center.x+left, rect.center.y+top);
		Mat M=image.clone();
		
		Mat cropped=new Mat();
		// get the rotation matrix
		M = org.opencv.imgproc.Imgproc.getRotationMatrix2D(rect.center, angle, 1.0);
		// perform the affine transformation on your image in src,
		// the result is the rotated image in rotated. I am doing
		// cubic interpolation here
		org.opencv.imgproc.Imgproc.warpAffine(src, rotated, M, src.size(), org.opencv.imgproc.Imgproc.INTER_LINEAR);
		
		// crop the resulting image, which is then given in cropped
		Imgproc.getRectSubPix(rotated, rect_size, rect.center, cropped);
		
		Imgproc.getRectSubPix(rotated, size, rect.center, result);
		
//		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(rotated),"Rotated");
//		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(cropped),"Cropped");
		
        
		return result;
	}
	
	
}
