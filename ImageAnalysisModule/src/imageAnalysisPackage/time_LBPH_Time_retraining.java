package imageAnalysisPackage;

import static org.bytedeco.javacpp.opencv_contrib.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_contrib.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;

import java.util.concurrent.ForkJoinPool;
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
public class time_LBPH_Time_retraining {
	
	static String datasetLabels [] = {"WITH_LIGHT"};//{"WITHOUT_LIGHT","WITH_LIGHT"};//{"WITH_LIGHT"};//,"WITHOUT_LIGHT"};
	
	//static int NUM_OF_RUNS=1;
	static int NUM_OF_PARTICIPANTS;

	final static Size outImageSize = new Size(500,750);

	
	public static void main(String args[]) throws IOException{
		run();
	}//end of main

	/**
	 * performing the analysis
	 * TODO:
	 * make it suitable for parallel computing
	 * https://humanoidreadable.wordpress.com/2014/12/31/forkjoin-nonrecursive-task/
	 */
	public static void run() throws IOException{
		for(String datasetLabel: datasetLabels){
			String trainingDir = "./DATASETS_NOISE_10000/training"+datasetLabel+"_noise/";//args[0];
			String testingDir="./DATASETS_NOISE_10000/testing"+datasetLabel+"_noise/";
			//create a writer to save the data
			File trainDir=new File(trainingDir);
			File[] trainingImages = trainDir.listFiles();
				
			File testDir=new File(testingDir);
			File[] testingImages = testDir.listFiles();
			
			NUM_OF_PARTICIPANTS=testingImages.length;
				
				
			System.out.println("Number of test images " + testingImages.length);
			FileWriter writer = new FileWriter("./DATASETS_NOISE_10000/results_LBPH_Retraining_10000_"+datasetLabel+".csv ");
				
			writer.append("number of participants");
			writer.append(',');
			writer.append("recognition rate");
		    writer.append(',');
		    writer.append("training time");
		    writer.append(',');
		    writer.append("average testing time");
		    writer.append(',');
		    writer.append("total matching time");
		    writer.append('\n');
		    writer.flush();
				
			//read files from training and testing sets:
				
			int numOfParticipants;
			//iterate over all 15 participants
			
			
			//list all training labels
			ArrayList<Integer> allLabels = new ArrayList<Integer>();
				
			for(File image : testingImages){
				int label = Integer.parseInt(image.getName().split("\\_")[0]);
				if(!allLabels.contains(label))
					allLabels.add(label);
				}
			System.out.println("NUMBER_OF_PARTICIPANTS "+allLabels.size());
			//randomly draw numOfParticipants labels from ArrayList of all labels
				
			Random  randomGenerator = new Random();
			
			
			//iterate over all participants
			for( numOfParticipants=2;numOfParticipants<=NUM_OF_PARTICIPANTS;numOfParticipants++){
				
				File resTestingDirectory=new File("./DATASETS_NOISE_10000/results"+datasetLabel);
				if(!resTestingDirectory.exists()){
					if(resTestingDirectory.mkdir())
						System.out.println(resTestingDirectory.getAbsolutePath()+" is created");
					else
						System.out.println("Failed to create a directory...");
				}

				double sumRecognitionRate = 0;
					
				int sumTestTime=0;
				int sumTrainTime=0;
				//repeat 10 times and take an average
				
				//for(int round =1;round<=NUM_OF_RUNS;round++){
				List<Integer> drawnTrainLabels = new ArrayList<Integer>();
						
				drawnTrainLabels=chooseRandomly(allLabels,numOfParticipants,randomGenerator);
						
				System.out.println("drawnTrainLabels  "+drawnTrainLabels.size());
				System.out.println("numOfParticipants "+numOfParticipants);
						

				
				
				
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
				
				for(int currentLabel: drawnTrainLabels){
					FileFilter fileFilter = new WildcardFileFilter(currentLabel+"*.jpg");
					File[] files = testDir.listFiles(fileFilter);
					
					if(files.length!=0){
						List<File> shuffledFilesForCurrentLabelArrayList = new ArrayList<>();
						for (File image : files){
							shuffledFilesForCurrentLabelArrayList.add(image);
						}
						Collections.shuffle(shuffledFilesForCurrentLabelArrayList);
						// now convert it back to array:
						
						File []shuffledFiles = (File[])shuffledFilesForCurrentLabelArrayList.toArray(new File[shuffledFilesForCurrentLabelArrayList.size()]);
//						System.out.println("Files for current label "+ currentLabel);
//						for (int i = 0; i < shuffledFiles.length; i++) {
//						   System.out.println(shuffledFiles[i]);
//						}
//						System.out.println("SELECTED   "+shuffledFiles[0]);
						newTestingImagesArrayList.add(shuffledFiles[0]);
					}
				}
						
				
								
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
					Mat img = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC3);
				          
				    //System.out.println(image.getPath());
				
			        img = org.bytedeco.javacpp.opencv_highgui.imread(image.getAbsolutePath(), org.bytedeco.javacpp.opencv_highgui.CV_LOAD_IMAGE_GRAYSCALE);
			        org.bytedeco.javacpp.opencv_imgproc.resize(img, img, new org.bytedeco.javacpp.opencv_core.Size(500,700));
	
			        int label = Integer.parseInt(image.getName().split("\\_")[0]);
				          
				          
				    images.put(counter, img);
				
				    labelsBuf.put(counter, label);
				
				    counter++;
				}//end of for
				
				    
//						      FaceRecognizer faceRecognizer = org.bytedeco.javacpp.opencv_contrib.createFisherFaceRecognizer();
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
			      FaceRecognizer faceRecognizer =createLBPHFaceRecognizer(6, 9, 9, 9, distTreshold);
				
				       
			      //long processingTime = System.currentTimeMillis()-startTime; 
			      faceRecognizer.set("threshold", 7500.0);//too big numbers
			      
			      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			      //measuring training time for current number of participants
			      long tempStartTrainingTime = System.currentTimeMillis();
				      
			      faceRecognizer.train(images, labels);
			      
			      long tempEndTrainingTime=System.currentTimeMillis()-tempStartTrainingTime;
			      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			      
			      int predictedLabel[] = new int[1];
			      double confidence[] = new double[1];
				      
				//		      
			      int testSetSize=testingImages.length;
			      int trainSetSize=testingImages.length;
				      
			      int[] matchingResults=new int[testSetSize];
			      int sum=0;
				      
			      int totalTestTime=0;
			      
			      System.out.println("start matching " +numOfParticipants+ " participants  to " +newTestingImages.length+" testing images");
			      //iterate over al testing images
			      int i=0;
			      
			      //to measure total time required to find all matchings for the selected instance 
			      long tempStartTotalMatchingTime = System.currentTimeMillis();
			      
			      for(File image: newTestingImages){
					    	
					  Mat testImage = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC3);
					  testImage= org.bytedeco.javacpp.opencv_highgui.imread(image.getAbsolutePath(), org.bytedeco.javacpp.opencv_highgui.CV_LOAD_IMAGE_GRAYSCALE);
					  org.bytedeco.javacpp.opencv_imgproc.resize(testImage, testImage, new org.bytedeco.javacpp.opencv_core.Size(500,700));
		
					  int actualLabel = Integer.parseInt(image.getName().split("\\_")[0]);
					  
					  
					  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					  //measuring testing time for current number of participants
					  long tempStartTestTime = System.currentTimeMillis();
					  faceRecognizer.predict(testImage, predictedLabel, confidence);
					  long tempEndTestTime=System.currentTimeMillis()-tempStartTestTime;
					  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					  
					  
					  if(predictedLabel[0]==actualLabel){
						  matchingResults[i]=1;
						  sum+=1;
					  }
					  else{
						  matchingResults[i]=0;
						  sum+=0;
					  }
					       
					  totalTestTime+=tempEndTestTime;
//					  System.out.println();
//					  System.out.println("Processing image "+image.getName());
//					  System.out.println("-----------------------------------------------------------------------------------");
//					  System.out.println("Recall Time: " + tempEndTestTime);
//					  System.out.println("Predicted label: " + predictedLabel[0]);
//					  System.out.println("Actual label: " + actualLabel);
//					  System.out.println("Prediction Confidence: " + confidence[0]);
//				      i++;
					 
					}//end of iterating over all testing images
				    
			      
			      	long tempEndTotalMatchingTime = System.currentTimeMillis()-tempStartTotalMatchingTime;
			      	System.out.println();
			    	System.out.println("-----------------------------------------------------------------------------------");
			    	System.out.println("End of processing");
				    System.out.println();
				    System.out.println("NUMBER OF INSTANCES TESTED "+newTestingImages.length);
				    System.out.println("Correctly recognized "+sum +" out of "+newTestingImages.length);
			    	System.out.println("RECOGNITION RATE: "+ (double)sum/newTestingImages.length);	
			    	System.out.println("TOTAL TRAIN TIME "+ tempEndTrainingTime);
			    	System.out.println("TOTAL TEST TIME "+ totalTestTime);
			    	System.out.println("TOTAL MATCHING TIME "+ totalTestTime);
			    	System.out.println("AVERAGE TEST TIME (RECALL)"+ (double)totalTestTime/newTestingImages.length);
			    	System.out.println("-----------------------------------------------------------------------------------");
			    	System.out.println();
			    	System.out.println();
			    	
			    	sumRecognitionRate+=(double)sum/newTestingImages.length;
					
					
					sumTrainTime+=tempEndTrainingTime;
					
						
						//SAVE TO FILE
	
				    	
				    sumTestTime+=totalTestTime/newTestingImages.length;

					
			    	//SAVE TO FILE
			    	writer.append(String.valueOf(numOfParticipants));
				    writer.append(',');
				    writer.append(String.valueOf((double)sum/newTestingImages.length));
				    writer.append(',');
				    writer.append(String.valueOf((double)tempEndTrainingTime));
				    writer.append(',');
				    writer.append(String.valueOf((double)totalTestTime/newTestingImages.length));
				    writer.append(',');
				    writer.append(String.valueOf((double)tempEndTotalMatchingTime));
				    writer.append('\n');
				    writer.flush();
				  
				}//end iterating over selected number of participants
				
				System.out.println("DONE PROCESING "+datasetLabel+".......................................................................");
			    writer.flush();
			    writer.close();
		}//end of iterating over all datasets
		
		
	}
	
	 /**
     * Create a new list which contains the specified number of elements from the source list, in a
     * random order but without repetitions.
     *
     * @param sourceList    the list from which to extract the elements.
     * @param itemsToSelect the number of items to select
     * @param random        the random number generator to use
     * @return a new list   containg the randomly selected elements
     */
    public static <T> List<T> chooseRandomly(List<T> sourceList, int itemsToSelect, Random random) {
        int sourceSize = sourceList.size();
 
        // Generate an array representing the element to select from 0... number of available
        // elements after previous elements have been selected.
        int[] selections = new int[itemsToSelect];
 
        // Simultaneously use the select indices table to generate the new result array
        ArrayList<T> resultArray = new ArrayList<T>();
        for (int count = 0; count < itemsToSelect; count++) {
        	// An element from the elements *not yet chosen* is selected
            int selection = random.nextInt(sourceSize - count);
            selections[count] = selection;
            // Store original selection in the original range 0.. number of available elements
 
            // This selection is converted into actual array space by iterating through the elements
            // already chosen.
            for (int scanIdx = count - 1; scanIdx >= 0; scanIdx--) {
                if (selection >= selections[scanIdx]) {
                    selection++;
                }
            }
            // When the first selected element record is reached all selections are in the range
            // 0.. number of available elements, and free of collisions with previous entries.
 
            // Write the actual array entry to the results
            resultArray.add(sourceList.get(selection));
        }
        return resultArray;
    }
	
	
}
