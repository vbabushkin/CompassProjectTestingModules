package imageAnalysisPackage;

import static org.bytedeco.javacpp.opencv_contrib.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_contrib.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;

import java.io.File;
import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.filefilter.WildcardFileFilter;
import org.bytedeco.javacpp.opencv_contrib.FaceRecognizer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_imgproc;
import org.opencv.core.Size;
//http://tech.thecoolblogs.com/2012/10/creating-lbph-local-binary-pattern.html
public class test_LBPH_Performance {
	
	static String datasetLabels [] = {"GRAY","RGB","GRAY_LIGHT","RGB_LIGHT"};
	
	static int NUM_OF_RUNS=1;
	static int NUM_OF_PARTICIPANTS=15;

	final static Size outImageSize = new Size(500,750);

	
	public static void main(String args[]) throws IOException{
		
		for(String datasetLabel: datasetLabels){
			
			 String trainingDir = "./training"+datasetLabel+"/";//args[0];
			 String testingDir="./testing"+datasetLabel+"/";
			 //create a writer to save the data
				
				
				FileWriter writer = new FileWriter("results"+datasetLabel+".csv ");
				
				writer.append("number of participants");
			    writer.append(',');
			    writer.append("recognition rate");
			    writer.append(',');
			    writer.append("training time");
			    writer.append(',');
			    writer.append("testing time");
			    writer.append('\n');
				
				//read files from training and testing sets:
				
				int numOfParticipants;
				//iterate over all 15 participants
				for( numOfParticipants=2;numOfParticipants<=NUM_OF_PARTICIPANTS;numOfParticipants++){
					FileWriter participantWriter = new FileWriter("./results"+datasetLabel+"/"+"participant_"+numOfParticipants+"_"+datasetLabel+".csv ");
					
					participantWriter.append("iteration #");
					participantWriter.append(',');
					participantWriter.append("recognition rate");
					participantWriter.append(',');
					participantWriter.append("training time");
					participantWriter.append(',');
					participantWriter.append("testing time");
					participantWriter.append('\n');
					double sumRecognitionRate = 0;
					
					int sumTestTime=0;
					int sumTrainTime=0;
					int numOfRuns=NUM_OF_RUNS;
					//repeat 10 times and take an average 
					for(int round =1;round<=numOfRuns;round++){
						
						File trainDir=new File(trainingDir);
						File[] trainingImages = trainDir.listFiles();
						
						File testDir=new File(testingDir);
						File[] testingImages = testDir.listFiles();
						
						//list all training labels
						ArrayList<Integer> allLabels = new ArrayList<Integer>();
						
						for(File image : trainingImages){
							int label = Integer.parseInt(image.getName().split("\\_")[0]);
							if(!allLabels.contains(label))
								allLabels.add(label);
						}
						
						//randomly draw numOfParticipants labels from ArrayList of all labels
						
						Random  randomGenerator = new Random();
						
						ArrayList<Integer> drawnTrainLabels = new ArrayList<Integer>();
						
						while(drawnTrainLabels.size()!=numOfParticipants){
							int index = randomGenerator.nextInt(allLabels.size());
					        int  randomParticipant = allLabels.get(index);
					        if(!drawnTrainLabels.contains(randomParticipant))
					        	drawnTrainLabels.add(randomParticipant);
						}
						
						//test
						for(int participant:drawnTrainLabels){
							System.out.println("participant number "+ participant+" is selected");
						}
						
						//now fill the array of selected training images
						List<File> newTrainingImagesArrayList=new ArrayList<File>();
						for(File image : trainingImages){
							int label = Integer.parseInt(image.getName().split("\\_")[0]);
							if(drawnTrainLabels.contains(label)){
								newTrainingImagesArrayList.add(image);
							}
						}
						
						
						
						//now fill the array of selected testing images
						List<File> newTestingImagesArrayList=new ArrayList<File>();
						
						
						//first shuffle testingImages
						
				//		List<File> shuffledTestingImagesArrayList = new ArrayList<>();
				//		for (File image : testingImages)
				//		{
				//			shuffledTestingImagesArrayList.add(image);
				//		}
				//		Collections.shuffle(shuffledTestingImagesArrayList);
				//		// now convert it back to array:
				//		
				//		File []shuffledTestingImages = (File[])shuffledTestingImagesArrayList.toArray(new File[shuffledTestingImagesArrayList.size()]);
						
						for(int currentLabel: drawnTrainLabels){
							FileFilter fileFilter = new WildcardFileFilter(currentLabel+"*.jpg");
							File[] files = testDir.listFiles(fileFilter);
							
							if(files.length!=0){
								List<File> shuffledFilesForCurrentLabelArrayList = new ArrayList<>();
								for (File image : files)
								{
									shuffledFilesForCurrentLabelArrayList.add(image);
								}
								Collections.shuffle(shuffledFilesForCurrentLabelArrayList);
								// now convert it back to array:
								
								File []shuffledFiles = (File[])shuffledFilesForCurrentLabelArrayList.toArray(new File[shuffledFilesForCurrentLabelArrayList.size()]);
								System.out.println("Files for current label "+ currentLabel);
								for (int i = 0; i < shuffledFiles.length; i++) {
								   System.out.println(shuffledFiles[i]);
								}
								System.out.println("SELECTED   "+shuffledFiles[0]);
								newTestingImagesArrayList.add(shuffledFiles[0]);
							}
						}
						
						
						
				//		for(File image : shuffledTestingImages){
				//			int label = Integer.parseInt(image.getName().split("\\_")[0]);
				//			
				//			
				//			if(drawnTrainLabels.contains(label) ){
				//				//add the topmost
				//				newTestingImagesArrayList.add(files[0]);
				//			}
				//		}
								
								
						File [] newTrainingImages = (File[])newTrainingImagesArrayList.toArray(new File[newTrainingImagesArrayList.size()]);
						
						File [] newTestingImages=(File[])newTestingImagesArrayList.toArray(new File[newTestingImagesArrayList.size()]);
						
						System.out.println();
						System.out.println("NUMBER OF TRAINING IMAGES "+newTrainingImages.length);
						System.out.println("NUMBER OF TEST IMAGES "+newTestingImages.length);
						System.out.println();
						
						MatVector images = new MatVector(newTrainingImages.length);
						Mat labels = new Mat(newTrainingImages.length, 1, CV_32SC1);
						IntBuffer labelsBuf = labels.getIntBuffer();
								
						int counter = 0;
				
						for (File image : newTrainingImages) {
				//			          Mat img = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC1);
				          Mat img = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC3);
				          
				          System.out.println(image.getPath());
				//			          img.createFrom(image);
				//			          ImageProcessingUtilities.displayImage(reconstructImage(image), "Testing "+ image.getPath());
				//			          
				          img = org.bytedeco.javacpp.opencv_highgui.imread(image.getAbsolutePath(), org.bytedeco.javacpp.opencv_highgui.CV_LOAD_IMAGE_GRAYSCALE);
				          org.bytedeco.javacpp.opencv_imgproc.resize(img, img, new org.bytedeco.javacpp.opencv_core.Size(500,700));
				          //opencv_imgproc.equalizeHist(img, img);
				          //org.bytedeco.javacpp.opencv_imgproc.threshold(img, img, 255,11, opencv_imgproc.ADAPTIVE_THRESH_GAUSSIAN_C);
				//			          opencv_imgproc.cvAdaptiveThreshold(img.asCvMat(), img.asCvMat(), 255, opencv_imgproc.CV_ADAPTIVE_THRESH_MEAN_C, opencv_imgproc.CV_THRESH_BINARY_INV, 5, 4);
				//			          ProcessImages.displayImage(img.getBufferedImage(), "Adaptive");
				          //org.bytedeco.javacpp.opencv_core.normalize(img, img);
				//			          Mat resizedTestImage = new Mat();
				//			          org.bytedeco.javacpp.opencv_core.Size newTestImageSize = new org.bytedeco.javacpp.opencv_core.Size(200,300); 
				//			          org.bytedeco.javacpp.opencv_imgproc.resize(img, resizedTestImage, newTestImageSize);
				          
				          int label = Integer.parseInt(image.getName().split("\\_")[0]);
				          
				          //System.out.println(image.getName());
				          
				          images.put(counter, img);
				
				          labelsBuf.put(counter, label);
				
				          counter++;
						}
				
					    long tempStartTrainingTime = System.currentTimeMillis();    
						      FaceRecognizer faceRecognizer = org.bytedeco.javacpp.opencv_contrib.createFisherFaceRecognizer();
//						      FaceRecognizer faceRecognizer = createEigenFaceRecognizer();// here we can put parameters -- threshold and number of eigenvectors
				//		      FaceRecognizer faceRecognizer = createEigenFaceRecognizer(15,5000.0);
				
					      
					      
				      // The following lines create an LBPH model for
				      // face recognition and train it with the images and
				      // labels read 
				      //
				      // The LBPHFaceRecognizer uses Extended Local Binary Patterns
				      // (it's probably configurable with other operators at a later
				      // point), and has the following default values
				      //
				      //      radius = 1
				      //      neighbors = 8
				      //      grid_x = 8
				      //      grid_y = 8
				      //
				      // So if you want a LBPH FaceRecognizer using a radius of
				      // 2 and 16 neighbors, call the factory method with:
				      //
				      //      createLBPHFaceRecognizer(2, 16);
				      //
				      // And if you want a threshold (e.g. 123.0) call it with its default values:
				      //
				      //      createLBPHFaceRecognizer(1,8,8,8,123.0)   
				      //by default
				      //FaceRecognizer faceRecognizer = createLBPHFaceRecognizer();
				      //6,9,9,9 gives the best recognition rate
					   
					  double distTreshold = 100; //distance threshold   
//				      FaceRecognizer faceRecognizer =createLBPHFaceRecognizer(6, 9, 9, 9, distTreshold);
				
				       
				      //long processingTime = System.currentTimeMillis()-startTime; 
				      faceRecognizer.set("threshold", 7500.0);//too big numbers
				
				      
				      faceRecognizer.train(images, labels);
				      
				      long tempEndTrainingTime=System.currentTimeMillis()-tempStartTrainingTime;
				      
				      int predictedLabel[] = new int[1];
				      double confidence[] = new double[1];
				      
				//		      
				      int testSetSize=testingImages.length;
				      int trainSetSize=testingImages.length;
				      
				      int[] matchingResults=new int[testSetSize];
				      int sum=0;
				      int totalTime=0;
				      
				      System.out.println("start matching .... ");
						
				      Random r = new Random();
				    
				 // generate a uniformly distributed int random numbers
				      
				    //int[] integers ={0,1,2,3,4,5,6,7,8,9,10,11,12};
				      
				    //int[] integers = new int[testSetSize];
				//		    for (int i = 0; i < integers.length; i++) {
				//		      integers[i] = r.nextInt(numOfClasses - 1) + 1;
				//		      System.out.println(integers[i]);
				//		    }
				//    
				//    
					    //for(int i:integers){
				    int i=0;
				    for(File image: newTestingImages){
					    	long tempStartTime = System.currentTimeMillis();
					    	//File image = testingImages[i];
				//			    	Mat testImage = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC1);
					    	Mat testImage = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC3);
					    	testImage= org.bytedeco.javacpp.opencv_highgui.imread(image.getAbsolutePath(), org.bytedeco.javacpp.opencv_highgui.CV_LOAD_IMAGE_GRAYSCALE);
					    	org.bytedeco.javacpp.opencv_imgproc.resize(testImage, testImage, new org.bytedeco.javacpp.opencv_core.Size(500,700));
				//			    	opencv_imgproc.equalizeHist(testImage, testImage);
					    	
					    	
				//			    	 opencv_imgproc.cvAdaptiveThreshold(testImage.asCvMat(), testImage.asCvMat(), 255, opencv_imgproc.CV_ADAPTIVE_THRESH_MEAN_C, opencv_imgproc.CV_THRESH_BINARY_INV, 5, 4);
				//			    	 ProcessImages.displayImage(testImage.getBufferedImage(), "Adaptive Testing");
					    	 //org.bytedeco.javacpp.opencv_core.normalize(testImage, testImage);
				//			        ImageProcessingUtilities.displayImage(reconstructImage(image), "Predicting  "+ image.getPath());
				//			    	
					        //System.out.println("ABSOLUTE PATH FOR TEST IMAGE: "+ testImageFiles[35].getAbsolutePath());
					        //int actualLabel = Integer.parseInt(testingImages[i].getName().split("\\_")[0]);
					    	
					    	
					        int actualLabel = Integer.parseInt(image.getName().split("\\_")[0]);
					        faceRecognizer.predict(testImage, predictedLabel, confidence);
					        
					        if(predictedLabel[0]==actualLabel){
					        	matchingResults[i]=1;
					        	sum+=1;
					        }
					        else{
					        	matchingResults[i]=0;
					        	sum+=0;
					        }
					        
					        
					        long tempEndTime=System.currentTimeMillis()-tempStartTime;
					//        
					        totalTime+=tempEndTime;
					        
					        
					        System.out.println();
					        System.out.println("Processing image "+image.getName());
					        System.out.println("-----------------------------------------------------------------------------------");
					        
					        System.out.println("Predicted label: " + predictedLabel[0]);
					        System.out.println("Actual label: " + actualLabel);
					        System.out.println("Prediction Confidence: " + confidence[0]);
					        i++;
						}
				    	System.out.println("-----------------------------------------------------------------------------------");
				    
				    	System.out.println("End of processing");
					    System.out.println();
					    System.out.println("Correctly recognized "+sum +" out of "+newTestingImages.length);
				    	System.out.println("RECOGNITION RATE: "+ (double)sum/newTestingImages.length);
				    	System.out.println();
				    	System.out.println("TOTAL TRAIN TIME "+ tempEndTrainingTime);
				    	System.out.println("TOTAL TEST TIME "+ totalTime);
				    	System.out.println("AVERAGE TEST TIME (RECALL)"+ (double)totalTime/newTestingImages.length);
				    	sumRecognitionRate+=(double)sum/newTestingImages.length;
						
						sumTestTime+=totalTime;
						sumTrainTime+=tempEndTrainingTime;
						
						
						//SAVE TO FILE
				    	participantWriter.append(String.valueOf(round));
				    	participantWriter.append(',');
				    	participantWriter.append(String.valueOf((double)sum/newTestingImages.length));
				    	participantWriter.append(',');
				    	participantWriter.append(String.valueOf(tempEndTrainingTime));
				    	participantWriter.append(',');
				    	participantWriter.append(String.valueOf((double)totalTime/newTestingImages.length));
				    	participantWriter.append('\n');
						
						
					}//end of for repeating 100 times
					participantWriter.flush();
					participantWriter.close();
					
					System.out.println("End of processing");
				    System.out.println();
				    System.out.println("AVERAGE VALUES OBTAINED");
			    	System.out.println("RECOGNITION RATE: "+ (double)sumRecognitionRate/numOfRuns);
			    	System.out.println();
			    	System.out.println("TOTAL TRAIN TIME "+ (double)sumTrainTime/numOfRuns);
			    	System.out.println("TOTAL TEST TIME "+ (double)sumTestTime/numOfRuns);
					
			    	//SAVE TO FILE
			    	writer.append(String.valueOf(numOfParticipants));
				    writer.append(',');
				    writer.append(String.valueOf((double)sumRecognitionRate/numOfRuns));
				    writer.append(',');
				    writer.append(String.valueOf((double)sumTrainTime/numOfRuns));
				    writer.append(',');
				    writer.append(String.valueOf((double)sumTestTime/numOfRuns));
				    writer.append('\n');
			    	
				  
				}//end iterating over 15 participants
				
				System.out.println("DONE PROCESING "+datasetLabel+".......................................................................");
			    writer.flush();
			    writer.close();
		}//end of iterating over all datasets
		
		
		
		
	}//end of main

}
