package videoProcessing;


import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.opencv_highgui.VideoCapture;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfInt;
import org.opencv.highgui.Highgui;


public class processVideo {
	//select folder
	//static String folderName ="1_GRAY_LIGHT_1";//"2_RGB_1";
	//select videoType -- outDepth, outIR, outRegistred, outVideo
	static String videoType = "outVideo";//"outRegistred"
	
	
	
	
	
	//static int frameIndex =80;
	/**
	 * 
	 * @param frameIndex
	 * @throws IOException
	 */
	public static void extractCurrentFrame(int frameIndex,String folderName) throws IOException{
		String initialImagePath=".//videos//"+folderName+"//";
		String savePath=".//videoFrames//";
		String currentFolder=folderName+"//"+videoType;
		
		//load library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME );
		File mainDir = new File(savePath+folderName);
		System.out.println(mainDir.getAbsolutePath());
		if(!mainDir.exists()){
			if(mainDir.mkdir())
				System.out.println(mainDir.getAbsolutePath()+" is created");
			else
				System.out.println("Failed to create a directory...");
		}
		
		String dirName = savePath+currentFolder;
		
		File directory = new File(dirName);
		System.out.println(directory.getAbsolutePath());
		if(!directory.exists()){
			if(directory.mkdir())
				System.out.println(directory.getPath()+" is created");
			else
				System.out.println("Failed to create a directory...");
		}
		
		String filename=initialImagePath+videoType+".mp4";
		VideoCapture capture=new VideoCapture();
		capture.open(filename);
		int length=(int) capture.get(7);
		System.out.println(length);
		Mat frame=new Mat();
		long startTime = System.currentTimeMillis();
		        
		int i=0;
		while(i<length){
		   capture.read(frame); //reads captured frame into the Mat image
		   System.out.println("Frame # " +i);
			   String outputName ="frame_"+videoType+"_"+i;
			   MatOfInt parameters = new MatOfInt();
			   parameters.fromArray(Highgui.CV_IMWRITE_JPEG_QUALITY, 100);
			   
			   
			   if(i==frameIndex)
				   org.opencv.highgui.Highgui.imwrite(savePath+currentFolder+"//"+ outputName +".jpg",BufferedImage2Mat(frame.getBufferedImage()),parameters);
			   //ImageIO.write(frame.getBufferedImage(), "JPG", new File(savePath+currentFolder+"//"+ outputName +".jpg"));
			   //saveByteData(frame.getBufferedImage(),outputName, savePath+currentFolder+"//");
			   i++;
		}
		long duration=System.currentTimeMillis()-startTime;  
		System.out.println("Time elapsed "+duration);
	}//end of main
	
	
	/**
	 * 
	 * @param image
	 * @param filename
	 */
	public static void saveByteData(BufferedImage image, String filename, String path){
		File dir = new File(path);
		if (!dir.exists()) {
			if (dir.mkdir()) {
				System.out.println(path + "     Directory is created!");
			} else {
				System.out.println("Failed to create directory!");
			}
		}
		try{
			byte [] b = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
	        FileOutputStream fos= new FileOutputStream(path+filename);
	        ObjectOutputStream oos= new ObjectOutputStream(fos);
	        oos.writeObject(b);
	        oos.close();
	        fos.close();
	      }catch(IOException ioe){
	           ioe.printStackTrace();
	      }
	}

	/**
	 * BufferedImage2Mat
	 * @param BufferedImage image
	 * @return Mat
	 */
	public static org.opencv.core.Mat BufferedImage2Mat(BufferedImage image){
		//source: http://stackoverflow.com/questions/18581633/fill-in-and-detect-contour-rectangles-in-java-opencv
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        org.opencv.core.Mat mat = new org.opencv.core.Mat(image.getHeight(),image.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, data);
        return mat;
    }
	
	
}
