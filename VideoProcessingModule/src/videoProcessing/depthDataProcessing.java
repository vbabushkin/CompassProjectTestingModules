package videoProcessing;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class depthDataProcessing {
	//select folder
	static String folderName = "1_GRAY_LIGHT_1";//"2_RGB_1";
	static String initialDataPath=".//videos//"+folderName+"//depthData//";
	static String initialVideoPath = ".//videoFrames//"+folderName+"//outVideo//";
	static int frameIndex = 60;
	private static Scanner inDepthDataFile;
	
	static int MIN_DEPTH_THRESHOLD = 515;//515;
	static int MAX_DEPTH_THRESHOLD = 540;//535;
			
	
	public static void main(String args[]) throws FileNotFoundException{
		//load library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME );
		
		String depthFrameFileName = initialDataPath +"outDepthByte_"+frameIndex;
		String  videoImageFileName = initialVideoPath+"frame_outVideo_"+frameIndex+".jpg";
		String  videoBackgroundFileName = initialVideoPath+"frame_outVideo_"+2+".jpg";
		
		//read files
		Mat rgbFrame = Highgui.imread(videoImageFileName, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		Mat rgbBackgFrame = Highgui.imread(videoBackgroundFileName,Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		
//		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(rgbFrame, new Size(960,540))),"segmented RGB Frame");
//		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(rgbBackgFrame, new Size(960,540))),"segmented RGB Frame");
	        
		Mat depthFrame = processDepthDataFile(depthFrameFileName, MIN_DEPTH_THRESHOLD, MAX_DEPTH_THRESHOLD);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(depthFrame), "Depth image");
		
		//depth background
		String depthBackgroundFileName = initialDataPath +"outDepthByte_"+2;
		Mat depthBackgroundFrame = processDepthDataFile(depthBackgroundFileName, MIN_DEPTH_THRESHOLD, MAX_DEPTH_THRESHOLD);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(depthBackgroundFrame), "Depth background image");
		
		//convert both images to grayscale and subtract backgrounds
		Imgproc.cvtColor(depthFrame, depthFrame, Imgproc.COLOR_RGB2GRAY);
		Imgproc.cvtColor(depthBackgroundFrame, depthBackgroundFrame, Imgproc.COLOR_RGB2GRAY);
		
		Mat depthFrameBackgroundSubtracted = new Mat();
		Core.subtract(depthBackgroundFrame,depthFrame, depthFrameBackgroundSubtracted);
		
	    Imgproc.threshold(depthFrameBackgroundSubtracted,depthFrameBackgroundSubtracted,0,255,Imgproc.THRESH_BINARY);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(depthFrameBackgroundSubtracted), "Depth image without background");
		
		
		Mat dst;
		
		
		Mat uncropped = depthFrameBackgroundSubtracted.clone();
		Rect roi = new Rect(10, 36, 490, 350);
		Mat cropped = new Mat(uncropped, roi);
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(cropped), "cropped");
		
		//this will be used as mask
		Mat finalDepthImage = new Mat(424, 512, cropped.type());
		
		cropped.copyTo(finalDepthImage.submat(roi));
		
		
		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(finalDepthImage), "finalDepthImage");
		
		//perform a homography:
	    
	    Mat hDepthFrameBckgSubtracted = videoProcessingUtilities.performHomographyTransformation(finalDepthImage, new Size(1920,1080));
		
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(hDepthFrameBckgSubtracted, new Size(960,540))),"Segmented Depth Frame after Homography");

	    //to fill the gaps
	    Imgproc.erode(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));
	    Imgproc.dilate(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));
//	    
//	    Imgproc.blur(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted, new Size(15,15));
	    
	    
	    Imgproc.threshold(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted,0, 255, Imgproc.THRESH_BINARY);
	    
	    Highgui.imwrite(initialDataPath +"hDepthFrameBckgSubtracted.jpg", hDepthFrameBckgSubtracted);
	    
	    
//		//Mat mask = hDepthFrameBckgSubtracted.clone();
//		//erode and dilate the mask to remove sharp edges
//	    //TODO:
//	    //check different cases, might not work for shoes with too deep pattern
//		Imgproc.erode(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));

//		Imgproc.dilate(hDepthFrameBckgSubtracted, hDepthFrameBckgSubtracted, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));

//		
//		ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(hDepthFrameBckgSubtracted, new Size(960,540))),"Segmented Depth Frame after Morphology");

		//Highgui.imwrite(initialDataPath +"mHDepthFrameBckgSubtracted.jpg", hDepthFrameBckgSubtracted);
		
		
		 MatOfPoint largestContour = videoProcessingUtilities.extractLargestContour(hDepthFrameBckgSubtracted.clone(),25000, 120000);
		 /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////   
	    //calculate convex hull
	    
	    ArrayList<MatOfPoint> hullContours = videoProcessingUtilities.calculateConvexHull(largestContour);
	    
	    Mat mask = videoProcessingUtilities.getContourMasked(hDepthFrameBckgSubtracted.clone(), hullContours.get(0));
	    
	    //Mat mask = videoProcessingUtilities.getContourMasked(hDepthFrameBckgSubtracted.clone(), largestContour);
	    
	    //draw the largest contour
	    videoProcessingUtilities.drawCurrentContour(hDepthFrameBckgSubtracted.clone(),   largestContour,"Segmented ");
	    
	    //draw the contour mask
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(mask, new Size(960,540))),"mask final ");
		  
	    //get segmented RGB image
	    //display
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(rgbFrame),"rgbFrame");
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(rgbBackgFrame),"rgbBackgFrame ");
	    
	    Mat segmentedRGBFrame = videoProcessingUtilities.segmentRGBImage(rgbFrame,rgbBackgFrame);
	    
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(segmentedRGBFrame, new Size(960,540))),"segmented RGB Frame");
        
        //then mask it with mask from depthFrame 
        Mat preMaskedRGBFrame= new Mat();
        segmentedRGBFrame.copyTo(preMaskedRGBFrame, mask); 
        //Imgproc.equalizeHist(preMaskedRGBFrame, preMaskedRGBFrame);
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(preMaskedRGBFrame, new Size(960,540))),"pre masked RGB Frame");
 
      //extract rectangle with masked shoe only
        //find bounding minarea rectangle corresponding to largest contour
        MatOfPoint2f pointsPreROI = new MatOfPoint2f(largestContour.toArray() ); 
        RotatedRect preROI = Imgproc.minAreaRect(pointsPreROI);
	    
        Mat preMaskedRGBFrameROI = videoProcessingUtilities.rotateExtractedShoeprint(preMaskedRGBFrame, preROI, new Size(500,750), 2);
        //get image containing the masked shoe only
        //Imgproc.getRectSubPix(preMaskedRGBFrame, preROI.size, preROI.center, preMaskedRGBFrameROI);
        
        
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(preMaskedRGBFrameROI),"preMaskedRGBFrameROI");
//       
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
	    Mat overlayed = videoProcessingUtilities.overlayImages(rgbFrame,hDepthFrameBckgSubtracted);

	    //segment overlayed image
	    Mat overlayedSegmented = new Mat();
	    Core.inRange(overlayed, new Scalar(215, 215, 215), new Scalar(230, 230, 230), overlayedSegmented);//215 and 225 for 1_20
	    
	    //Core.inRange(overlayed, new Scalar(220, 220, 220), new Scalar(225, 225, 225), overlayedSegmented);
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(overlayedSegmented, new Size(960,540))),"Segmented OVERLAYED Frame # ");
	    
	    //Highgui.imwrite("./images/Increase.jpg",overlayedSegmented);
	    
	    ///////////////////////////////////////////////////////////////////////////////////////////////////////
	    //now find contours, hulls and extract shoes
	    //check if shoelike object is detected and return detected object's contours as ArrayList
	   if( videoProcessingUtilities.shoeLikeObjectDetected(overlayedSegmented.clone(), 25000, 120000)){
		   System.out.println("DETECTED ");
	   }
	    
	   
	   //Imgproc.cvtColor(overlayedSegmented, overlayedSegmented, Imgproc.COLOR_BGR2GRAY);
	   
	   Mat copyOfhDepthFrameBckgSubtracted =videoProcessingUtilities.removeHolesFromContour( overlayedSegmented.clone());
	   
	   ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(copyOfhDepthFrameBckgSubtracted, new Size(960,540))),"gaps removed ");
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
	 	   	videoProcessingUtilities.drawCurrentContour(copyOfhDepthFrameBckgSubtracted.clone(), approxMatOfPoint,"best contour");
	 	   		}
	 		}
	   
	 		System.out.println("SIZE OF RESULTING CONTOURS: "+resContours.size());
	 		
	   
		//recognize largest contours
	   MatOfPoint finalMaxContour = videoProcessingUtilities.extractLargestContour(copyOfhDepthFrameBckgSubtracted.clone(),25000, 120000);
	    
	    //calculate convex hull
	    
	    ArrayList<MatOfPoint> finalHullContours = videoProcessingUtilities.calculateConvexHull(finalMaxContour);
	    Imgproc.cvtColor(copyOfhDepthFrameBckgSubtracted, copyOfhDepthFrameBckgSubtracted, Imgproc.COLOR_GRAY2BGR);
	    Imgproc.drawContours(copyOfhDepthFrameBckgSubtracted, finalHullContours, -1, new Scalar(0,255,0), 2);
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(copyOfhDepthFrameBckgSubtracted, new Size(960,540))),"gaps removed ");
		   
	    //drawHullContours(copyOfhDepthFrameBckgSubtracted, finalMaxContour);
	   ///////////////////////////////////////////////////////////////////////////////////////////////////////
	    MatOfPoint2f finalMaskedPointsROI = new MatOfPoint2f(finalHullContours.get(0).toArray() ); 
        RotatedRect finalROI = Imgproc.minAreaRect(finalMaskedPointsROI);
        //Mat finalMaskedRGBFrameROI = rotateExtractedShoeprint(preMaskedRGBFrame, finalROI, new Size(500,750), 2);
        //get image containing the masked shoe only
        //Imgproc.getRectSubPix(preMaskedRGBFrame, preROI.size, preROI.center, preMaskedRGBFrameROI);
        
        
//        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(finalMaskedRGBFrameROI),"FINALLY");
        
	    Mat newMask = videoProcessingUtilities.getContourMasked(copyOfhDepthFrameBckgSubtracted,finalHullContours.get(0));
	    
	    
	    ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(videoProcessingUtilities.resizeImage(newMask, new Size(960,540))),"mask");
//	    //mask registred  
        Mat imageROIRegistred= new Mat();
//        //Imgproc.cvtColor(preMaskedRGBFrame, preMaskedRGBFrame, Imgproc.COLOR_BGR2GRAY);
        preMaskedRGBFrame.copyTo(imageROIRegistred, newMask); 
//	    
        Mat MaskedRGBFrameROI = videoProcessingUtilities.rotateExtractedShoeprint(imageROIRegistred, finalROI, new Size(500,750), 2);
//           
        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(MaskedRGBFrameROI),"FINALLY");
        
        Highgui.imwrite(initialDataPath +"MaskedRGBFrameROI_"+frameIndex+".jpg", MaskedRGBFrameROI);
//	    
//        Imgproc.adaptiveThreshold(MaskedRGBFrameROI, MaskedRGBFrameROI, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
//        ProcessImages.displayImage(ProcessImages.Mat2BufferedImage(MaskedRGBFrameROI),"ADAPTIVE THRESHOLD");
	    
	   System.out.println("Done..........");
	    
	   System.out.println("pixelCount "+pixelCount);
	   System.out.println("AVERAGE PIXEL VALUE "+ pixelCount/(preMaskedRGBFrameROI.width()*preMaskedRGBFrameROI.height()));
	   System.out.println("Shoe Area "+ Imgproc.contourArea(finalMaxContour));
	    

	}//END OF MAIN
	
	
	/**
	 * converts depth data to opencv Mat object leaving depth values that are only within min and max thresholds
	 * @param path
	 * @param minThreshold
	 * @param maxThreshold
	 * @return
	 * @throws FileNotFoundException
	 */
	public static Mat processDepthDataFile(String path,int minThreshold, int maxThreshold) throws FileNotFoundException{
		File depthData = new File(path);
		
		double[][]depthDataArray = new double[1][217088];
		
		//read depth data into array
		int count = 0;
		
		inDepthDataFile = new Scanner(depthData);//.useDelimiter(",\\s*");

		while(inDepthDataFile.hasNext()){
			String currentStr=inDepthDataFile.nextLine();
			if(!currentStr.isEmpty())
				depthDataArray[0][count++] = Double.parseDouble(currentStr);
		}
		
		double depthDataMatrix[][] = new double [512][424];
		
		depthDataMatrix= reshape(depthDataArray,512,424);
		
		Mat matDepthDataMatrix = new Mat(512,424, CvType.CV_64F);
		
		
		//cut-off the remaining depth values
		for(int i = 0;i<depthDataMatrix.length;i++){
            for(int j = 0;j<depthDataMatrix[0].length;j++){
            	if(depthDataMatrix[i][j]>maxThreshold || depthDataMatrix[i][j]<minThreshold)
            		depthDataMatrix[i][j]=0;
            }
		}
		
		
		
		//find max value
		double max = 0;
		
		for(int i = 0;i<depthDataMatrix.length;i++){
            for(int j = 0;j<depthDataMatrix[0].length;j++){
            	if(depthDataMatrix[i][j]>max)
            		max=depthDataMatrix[i][j];
            }
		}
		
		
		//FILL THE DEPTH MATRIX
		//System.out.println("Max Element "+ max);
		
		for(int i = 0;i<depthDataMatrix.length;i++){
		    for(int j = 0;j<depthDataMatrix[0].length;j++){
		    	matDepthDataMatrix.put(i, j, depthDataMatrix[i][j]/max*255.0);
		    }
		}
		
//		//printout the depth matrix
//		for(int i = 0;i<depthDataMatrix.length;i++){
//		    for(int j = 0;j<depthDataMatrix[0].length;j++){
//		        System.out.print(depthDataMatrix[i][j]+"\t");
//		    }
//        System.out.println();
//		}
//		
		
		//apply colormap to visualize
		Mat processedMathDepthImage= new Mat(matDepthDataMatrix.size(),CvType.CV_8U);
		matDepthDataMatrix.convertTo(processedMathDepthImage,CvType.CV_8UC1);
		
		Core.transpose(processedMathDepthImage, processedMathDepthImage);
		org.opencv.contrib.Contrib.applyColorMap(processedMathDepthImage,processedMathDepthImage,org.opencv.contrib.Contrib.COLORMAP_JET);

		return processedMathDepthImage;
	}
	
	/**
	 * reshaping for depthDataArray to make it 512X424 
	 * @param A
	 * @param m
	 * @param n
	 * @return
	 */
	public static double[][] reshape(double[][] A, int m, int n) {
        int origM = A.length;
        int origN = A[0].length;
        if(origM*origN != m*n){
            throw new IllegalArgumentException("New matrix must be of same area as matix A");
        }
        double[][] B = new double[m][n];
        double[] A1D = new double[A.length * A[0].length];

        int index = 0;
        for(int i = 0;i<A.length;i++){
            for(int j = 0;j<A[0].length;j++){
                A1D[index++] = A[i][j];
            }
        }

        index = 0;
        for(int i = 0;i<n;i++){
            for(int j = 0;j<m;j++){
                B[j][i] = A1D[index++];
            }

        }
        return B;
    }
}
