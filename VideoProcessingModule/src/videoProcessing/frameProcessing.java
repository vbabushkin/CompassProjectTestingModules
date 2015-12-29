package videoProcessing;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class frameProcessing {
	static int MIN_AREA=50000;
	static int MAX_AREA=150000;
	
	//select folder
	static String folderName = "1_GRAY_LIGHT_1";//"2_RGB_1";
	static String initialDataPath=".//videos//"+folderName+"//depthData//";
	static String initialVideoPath = ".//videoFrames//"+folderName+"//outVideo//";
	
	static int frameIndex = 70;
	
	private static Scanner inDepthDataFile;
	
	static int MIN_DEPTH_THRESHOLD = 515;//515;
	static int MAX_DEPTH_THRESHOLD = 545;//535;
	
	public static void main(String args[]) throws IOException{
		//load library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME );
		
		processVideo.extractCurrentFrame(frameIndex, folderName);
		
		String depthFrameFileName = initialDataPath +"outDepthByte_"+frameIndex;
		String  videoImageFileName = initialVideoPath+"frame_outVideo_"+frameIndex+".jpg";
		String  videoBackgroundFileName = initialVideoPath+"frame_outVideo_"+2+".jpg";
		String depthBackgroundFileName = initialDataPath +"outDepthByte_"+2;
		
		//read files
		Mat rgbFrame = Highgui.imread(videoImageFileName, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		Mat rgbBackgFrame = Highgui.imread(videoBackgroundFileName,Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		Mat depthFrame = depthDataProcessing.processDepthDataFile(depthFrameFileName, MIN_DEPTH_THRESHOLD, MAX_DEPTH_THRESHOLD);
		
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(rgbFrame, new Size(960,540))),"rgbFrame");
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(rgbBackgFrame, new Size(960,540))),"rgbBackgFrame");
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(depthFrame), "Depth image");
		
		//extract depth background
		Mat depthBackgroundFrame = depthDataProcessing.processDepthDataFile(depthBackgroundFileName, MIN_DEPTH_THRESHOLD, MAX_DEPTH_THRESHOLD);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(depthBackgroundFrame), "Depth background image");
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//convert both images to grayscale and subtract backgrounds
		Imgproc.cvtColor(depthFrame, depthFrame, Imgproc.COLOR_RGB2GRAY);
		Imgproc.cvtColor(depthBackgroundFrame, depthBackgroundFrame, Imgproc.COLOR_RGB2GRAY);
		
		Mat depthFrameBackgroundSubtracted = new Mat();
		Core.subtract(depthBackgroundFrame,depthFrame, depthFrameBackgroundSubtracted);
		
	    Imgproc.threshold(depthFrameBackgroundSubtracted,depthFrameBackgroundSubtracted,0,255,Imgproc.THRESH_BINARY);
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		Mat uncropped = depthFrameBackgroundSubtracted.clone();
		Rect roi = new Rect(10, 36, 490, 350);
		
		Mat cropped = cropDepthImage(uncropped, roi);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(cropped), "Depth image without background cropped");
		
		Mat finalDepthImage = cropped.clone();
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		//perform a homography:
	    Mat hDepthFrameBckgSubtracted = videoProcessingUtilities.performHomographyTransformation(finalDepthImage, new Size(1920,1080));
		
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(hDepthFrameBckgSubtracted, new Size(960,540))),"Segmented Depth Frame after Homography");

	    //to fill the gaps
	    hDepthFrameBckgSubtracted = applyMorphology(hDepthFrameBckgSubtracted,30,new Size(3,3));
	    
	    //apply threshold
	    Imgproc.threshold(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted,0, 255, Imgproc.THRESH_BINARY);
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(hDepthFrameBckgSubtracted, new Size(960,540))),"Segmented Depth Frame after erosion and dilation");
	    
	    
		//extract all detected contours
	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    List<MatOfPoint> largestContours = videoProcessingUtilities.extractAllDetectedContours(hDepthFrameBckgSubtracted, MIN_AREA, MAX_AREA);
		
	    System.out.println("Found contours "+largestContours.size());
	    //draw contours
		int i=0;
		for(MatOfPoint largestContour : largestContours){
			videoProcessingUtilities.drawCurrentContour(hDepthFrameBckgSubtracted, largestContour, "contour #"+i);
			//calculate convex hull
		    ArrayList<MatOfPoint> hullContours = videoProcessingUtilities.calculateConvexHull(largestContour);
		    videoProcessingUtilities.drawHullContours(hDepthFrameBckgSubtracted, largestContour);
		    Mat mask = videoProcessingUtilities.getContourMasked(hDepthFrameBckgSubtracted.clone(), hullContours.get(0));
		    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(mask, new Size(960,540))),"mask #"+i);
		    
			i++;
		}
	    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		
		

	}//end of main
	/**
	 * 
	 * @param image
	 * @param numberOfTimes
	 * @param kernel
	 * @return
	 */
	static Mat applyMorphology(Mat image, int numberOfTimes, Size kernel){
		for(int i=0;i<numberOfTimes;i++)
			Imgproc.erode(image, image, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernel));
		for(int i=0;i<numberOfTimes;i++)
			Imgproc.dilate(image, image, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernel));
		return image;
	}
	/**
	 * 
	 * @param uncropped
	 * @param roi
	 * @return
	 */
	static Mat cropDepthImage(Mat uncropped, Rect roi){
		
		Mat cropped = new Mat(uncropped, roi);
		//ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(cropped), "cropped");
		
		//this will be used as mask
		Mat finalDepthImage = new Mat(424, 512, cropped.type());
		
		cropped.copyTo(finalDepthImage.submat(roi));
		return cropped;
	}
	
//	Mat maskDepthImage(Mat rgbFrame){
//		//then mask it with mask from depthFrame 
//        Mat preMaskedRGBFrame= new Mat();
//        segmentedRGBFrame.copyTo(preMaskedRGBFrame, mask); 
//        //Imgproc.equalizeHist(preMaskedRGBFrame, preMaskedRGBFrame);
//        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(preMaskedRGBFrame, new Size(960,540))),"pre masked RGB Frame");
// 
//	}
		

}
