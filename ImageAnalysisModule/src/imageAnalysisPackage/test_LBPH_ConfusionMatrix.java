package imageAnalysisPackage;

import static org.bytedeco.javacpp.opencv_contrib.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.bytedeco.javacpp.opencv_contrib.FaceRecognizer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.opencv.core.Size;

public class test_LBPH_ConfusionMatrix {
	static String datasetLabels [] = {"WITH_LIGHT"};//{"WITH_LIGHT"};//,"WITHOUT_LIGHT"};
	
	static int NUM_OF_RUNS=1;
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
		double sumRecognitionRate = 0;
		//iterate over all dataset labels
		String datasetLabel=datasetLabels[0];

		String trainingDir = "./DATASETS_COMBINED/train_"+datasetLabel;//+"_noise/";//args[0];
		String testingDir="./DATASETS_COMBINED/test_"+datasetLabel;//+"_noise/";
	
		File trainDir=new File(trainingDir);
		File[] trainingImages = trainDir.listFiles();
			
		File testDir=new File(testingDir);
		File[] testingImages = testDir.listFiles();
		
		NUM_OF_PARTICIPANTS=trainingImages.length/2;
		
		
		System.out.println("Number of train images " + trainingImages.length);
		System.out.println("Number of test images " + testingImages.length);
		
		int numOfParticipants=NUM_OF_PARTICIPANTS;
		//iterate over all 15 participants
		
		
		//list all training labels
		ArrayList<Integer> allLabels = new ArrayList<Integer>();
			
		for(File image : testingImages){
			int label = Integer.parseInt(image.getName().split("\\_")[0]);
			if(!allLabels.contains(label))
				allLabels.add(label);
			}
		
		//randomly draw numOfParticipants labels from ArrayList of all labels
		Random  randomGenerator = new Random();
		
		List<Integer> drawnTrainLabels = new ArrayList<Integer>();
		
		drawnTrainLabels=chooseRandomly(allLabels,numOfParticipants,randomGenerator);
		
		System.out.println("drawnTrainLabels  "+drawnTrainLabels.size());
		System.out.println("numOfParticipants "+numOfParticipants);
		
		
		//now fill the array of selected training images
		List<File> newTrainingImagesArrayList=selectTrainingImages(trainingImages, drawnTrainLabels );
		
		//now fill the array of selected testing images
		List<File> newTestingImagesArrayList=selectTestingImages(testingImages,drawnTrainLabels);
		File [] newTrainingImages = (File[])newTrainingImagesArrayList.toArray(new File[newTrainingImagesArrayList.size()]);
		
		File [] newTestingImages=(File[])newTestingImagesArrayList.toArray(new File[newTestingImagesArrayList.size()]);
		
		System.out.println();
		System.out.println("NUMBER OF TRAINING IMAGES "+newTrainingImages.length);
		System.out.println("NUMBER OF TEST IMAGES "+newTestingImages.length);
		System.out.println();
		
		System.out.println("TRAINING SET");
		for(File trainImage:newTrainingImages){
			System.out.println(trainImage.getName());
		}
		
		System.out.println();
		System.out.println("TESTING SET");
		for(File testImage:newTestingImages){
			System.out.println(testImage.getName());
		}
		
		MatVector images = new MatVector(newTrainingImages.length);
		Mat labels = new Mat(newTrainingImages.length, 1, CV_32SC1);
		IntBuffer labelsBuf = labels.getIntBuffer();
				
		int counter = 0;

		for (File image : newTrainingImages) {

			Mat img = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC3);
               
			img = org.bytedeco.javacpp.opencv_highgui.imread(image.getAbsolutePath(), org.bytedeco.javacpp.opencv_highgui.CV_LOAD_IMAGE_GRAYSCALE);
			org.bytedeco.javacpp.opencv_imgproc.resize(img, img, new org.bytedeco.javacpp.opencv_core.Size(500,700));
          
			int label = Integer.parseInt(image.getName().split("\\_")[0]);
          
			images.put(counter, img);

			labelsBuf.put(counter, label);

			counter++;
          
		}//end of iterating over all training images
		
		 long tempStartTrainingTime = System.currentTimeMillis();  
		
		 double distTreshold = 100; //distance threshold   
		 //create face recognizer
	      FaceRecognizer faceRecognizer =createLBPHFaceRecognizer(6, 9, 9, 9, distTreshold);
	
	       
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
	      
	      int totalTestTime=0;
	      
	      System.out.println("start matching .... ");
		
	      
	      //now test the model on the dataset
	     // TODO:to function
		
	      
	      
	      ArrayList<Integer> testLabels = new ArrayList<>();
	      Map<String, Integer> confusionMatrix = new HashMap<String, Integer>();
	      
	      //fill the newly created confusion matrix with zeros
	      for(int row = 0;row<drawnTrainLabels.size();row++)
	    	  for(int col=0; col< drawnTrainLabels.size(); col++){
	    		  confusionMatrix.put(new String(drawnTrainLabels.get(row)+","+drawnTrainLabels.get(col)), 0);
	    	  }
	    		  
	      
	      int i=0;
	      for(int k=0;k<newTestingImages.length;k++){
//		  for(File image: newTestingImages){
	    	  File image=newTestingImages[k];
	    	  
			  	long tempStartTestTime = System.currentTimeMillis();
		    	//File image = testingImages[i];
	//			    	Mat testImage = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC1);
		    	Mat testImage = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC3);
		    	testImage= org.bytedeco.javacpp.opencv_highgui.imread(image.getAbsolutePath(), org.bytedeco.javacpp.opencv_highgui.CV_LOAD_IMAGE_GRAYSCALE);
		    	org.bytedeco.javacpp.opencv_imgproc.resize(testImage, testImage, new org.bytedeco.javacpp.opencv_core.Size(500,700));
	
		        int actualLabel = Integer.parseInt(image.getName().split("\\_")[0]);
		        
		        if(!testLabels.contains(actualLabel))
		        	testLabels.add(actualLabel);
		        
		        
		        faceRecognizer.predict(testImage, predictedLabel, confidence);
		        
		        System.out.println(new String(actualLabel+","+predictedLabel[0]));
		        int matchingScore=confusionMatrix.get(new String(actualLabel+","+predictedLabel[0]));
		        matchingScore+=1;
	        	confusionMatrix.put(new String(actualLabel+","+predictedLabel[0]), matchingScore);
		        
		        if(predictedLabel[0]==actualLabel){
		        	matchingResults[i]=1;
		        	sum+=1;
		        	
		        }
		        else{
		        	matchingResults[i]=0;
		        	sum+=0;
		        }
		        
		        
		        long tempEndTestTime=System.currentTimeMillis()-tempStartTestTime;
		        
		        totalTestTime+=tempEndTestTime;
		        
		        
		        System.out.println();
		        System.out.println("Processing image "+image.getName());
		        System.out.println("-----------------------------------------------------------------------------------");
		        System.out.println("Recall Time: " + tempEndTestTime);
		        System.out.println("Predicted label: " + predictedLabel[0]);
		        System.out.println("Actual label: " + actualLabel);
		        System.out.println("Prediction Confidence: " + confidence[0]);
		        i++;
			}//end of iterating over all testing images
	      
		    System.out.println("-----------------------------------------------------------------------------------");
		    
	    	System.out.println("End of processing");
		    System.out.println();
		    System.out.println("Correctly recognized "+sum +" out of "+newTestingImages.length);
	    	System.out.println("RECOGNITION RATE: "+ (double)sum/newTestingImages.length);
	    	System.out.println();
	    	System.out.println("TOTAL TRAIN TIME "+ tempEndTrainingTime);
	    	System.out.println("TOTAL TEST TIME "+ totalTestTime);
	    	System.out.println("AVERAGE TEST TIME (RECALL)"+ (double)totalTestTime/newTestingImages.length);
	    	sumRecognitionRate+=(double)sum/newTestingImages.length;

//		      for(int row = 0;row<drawnTrainLabels.size();row++)
//		    	  for(int col=0; col< drawnTrainLabels.size(); col++){
//		    		  System.out.print(new String(drawnTrainLabels.get(row)+","+drawnTrainLabels.get(col))+"  "+confusionMatrix.get(new String(drawnTrainLabels.get(row)+","+drawnTrainLabels.get(col)))+"\n" );
//		    	  }
	      
	    	//fill and printout confusion matrix
	    	int confusionMatrixValues[][]= new int[drawnTrainLabels.size()][drawnTrainLabels.size()];
	    	//to store labels and their coefficients
	    	ArrayList<Integer> confusionMatrixLabels= new ArrayList<Integer>();
	    	
	    	  System.out.println();
		      System.out.println("CONFUSION MATRIX: ");
		      System.out.println();
		      for(int row = 0;row<drawnTrainLabels.size();row++)
		    	  System.out.print("\t"+drawnTrainLabels.get(row));
		      System.out.println();
		      for(int row = 0;row<drawnTrainLabels.size();row++){
		    	  System.out.print(drawnTrainLabels.get(row));
		    	  confusionMatrixLabels.add(drawnTrainLabels.get(row));
		    	  for(int col=0; col< drawnTrainLabels.size(); col++){
		    		  confusionMatrixValues[row][col]=confusionMatrix.get(new String(drawnTrainLabels.get(row)+","+drawnTrainLabels.get(col)));
		    		  System.out.print("\t"+confusionMatrix.get(new String(drawnTrainLabels.get(row)+","+drawnTrainLabels.get(col))));
		    	  }
		    	  System.out.println();
		      }
		    	
		      
		      System.out.println();
		      System.out.println("TEST");
		      for(int lab:confusionMatrixLabels)
		    	  System.out.println(lab+ "  index  "+confusionMatrixLabels.indexOf(lab) );
		      System.out.println();
		      
		      
		      
		      double precisionArray []= new double[drawnTrainLabels.size()];
		      double recallArray []= new double[drawnTrainLabels.size()];
		      double F1scoreArray []= new double[drawnTrainLabels.size()];
		      
		      int truePositives [] = new int[drawnTrainLabels.size()];
		      int falsePositives [] = new int[drawnTrainLabels.size()];
		      int falseNegatives [] = new int[drawnTrainLabels.size()];
		    
		      //fill array of true positives and false negatives for each label
		      for(int row=0; row <drawnTrainLabels.size();row++)
		    	  for(int col=0; col< drawnTrainLabels.size(); col++){
		    		  if(row==col){
		    			  truePositives[row]=confusionMatrixValues[row][col];
		    			  falseNegatives[row]+=0;
		    		  }
		    		  
		    		  else{
		    			  falseNegatives[row]+=confusionMatrixValues[row][col];
		    		  }
		    	  }
		      
		      //Fill array with false positives for each label
		      for(int col=0; col< drawnTrainLabels.size(); col++){
		    	  for(int row=0; row <drawnTrainLabels.size();row++){
		    		  if(col==row){
		    			  falsePositives[col]=0;
		    		  }
		    		  
		    		  else{
		    			  falsePositives[col]+=confusionMatrixValues[row][col];
		    		  }
		    	  }
		      }
		      
		      double avgPrecision =0;
		      double avgRecall = 0;
		      double avgF1score=0;
		      //calculate precision, recall and f1 score
		      for(int row=0;row <drawnTrainLabels.size();row++){
		    	  precisionArray[row]=(truePositives[row]/(double)(truePositives[row]+falsePositives[row]));
		    	  if((truePositives[row]+falsePositives[row])==0)
		    		  precisionArray[row]=0;
		    	  avgPrecision+=precisionArray[row];
		    	  recallArray[row]=(truePositives[row]/(double)(truePositives[row]+falseNegatives[row]));
		    	  if((truePositives[row]+falseNegatives[row])==0)
		    		  recallArray[row]=0;
		    	  avgRecall+=recallArray[row];
		    	  F1scoreArray[row]=2*(precisionArray[row]*recallArray[row]/(double)(precisionArray[row]+recallArray[row]));
		    	  if((precisionArray[row]+recallArray[row])==0)
		    		  F1scoreArray[row]=0;
		    	  avgF1score+=F1scoreArray[row];
		    		  
		      }
		      
		      avgPrecision=avgPrecision/numOfParticipants;
		      avgRecall=avgRecall/numOfParticipants;
		      avgF1score=avgF1score/numOfParticipants;
		      
		      //to test
		      System.out.println("TRUE POSITIVES");
		      for(int row=0;row <drawnTrainLabels.size();row++)
		    	  System.out.println(truePositives[row]);
		      
		      System.out.println("FALSE NEGATIVES");
		      for(int row=0;row <drawnTrainLabels.size();row++)
		    	  System.out.println(falseNegatives[row]);
		      
		      System.out.println("FALSE POSITIVES");
		      for(int row=0;row <drawnTrainLabels.size();row++)
		    	  System.out.println(falsePositives[row]);
		      
		      System.out.println("PRECISION");
		      for(int row=0;row <drawnTrainLabels.size();row++)
		    	  System.out.println(precisionArray[row]);
		      
		      System.out.println("RECALL");
		      for(int row=0;row <drawnTrainLabels.size();row++)
		    	  System.out.println(recallArray[row]);
		      
		      System.out.println("F1 SCORE");
		      for(int row=0;row <drawnTrainLabels.size();row++)
		    	  System.out.println(F1scoreArray[row]);
		      
		      
		      System.out.println("Average precison "+ avgPrecision);
		      System.out.println("Average recall "+  avgRecall);
		      System.out.println("Average F1 SCORE "+ avgF1score);
	}//end of run
	
	/**
	 * 
	 * @param trainingImages
	 * @param drawnTrainLabels
	 * @return
	 */
	static List<File>selectTrainingImages(File[] trainingImages,List<Integer> drawnTrainLabels ){
		List<File> newTrainingImagesArrayList=new ArrayList<File>();
		for(File image : trainingImages){
			int label = Integer.parseInt(image.getName().split("\\_")[0]);
			if(drawnTrainLabels.contains(label)){
				newTrainingImagesArrayList.add(image);
			}
		}
		return newTrainingImagesArrayList;
	}
	
	/**
	 * 
	 * @param testingImages
	 * @param drawnTrainLabels
	 * @return
	 */
	static List<File>selectTestingImages(File[] testingImages,List<Integer> drawnTrainLabels ){
		List<File> newTestingImagesArrayList=new ArrayList<File>();

		for(File image : testingImages){
			int label = Integer.parseInt(image.getName().split("\\_")[0]);
			if(drawnTrainLabels.contains(label)){
				newTestingImagesArrayList.add(image);
			}
		}
		return newTestingImagesArrayList;
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
    }//end of chooseRandomly
	
	
    /**
     * 	
     * @param testingImages
     * @param faceRecognizer
     * @param numberOfParticipants
     */
    public static double[] testOnDataset(File[] testingImages,FaceRecognizer faceRecognizer, int numberOfParticipants){
    	double result[]=new double[2];
    	int predictedLabel[] = new int[1];
    	double confidence[] = new double[1];
    	
    	//	      
    	int testSetSize=testingImages.length;
    	
    	
    	int[] matchingResults=new int[testSetSize];
    	int sum=0;
    	int totalTestTime=0;
    	int i=0;

    	for(File image:testingImages){
    		long tempStartTestTime = System.currentTimeMillis();
    		Mat testImage = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC3);
    	
    		testImage = org.bytedeco.javacpp.opencv_highgui.imread(image.getAbsolutePath(), org.bytedeco.javacpp.opencv_highgui.CV_LOAD_IMAGE_GRAYSCALE);
    	
    	
    		org.bytedeco.javacpp.opencv_imgproc.resize(testImage, testImage, new org.bytedeco.javacpp.opencv_core.Size(500,750));
    	
//    		if(equalizeHistograms)
//    			opencv_imgproc.equalizeHist(testImage, testImage);
//    	
    	
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
    	
    	
    		long tempEndTestTime=System.currentTimeMillis()-tempStartTestTime;
    		//        
    		totalTestTime+=tempEndTestTime;

    		i++;
    	}
    	System.out.println();
    	
    	
    	
    	System.out.println("Correctly recognized "+sum +" out of "+numberOfParticipants);
    	System.out.println("RECOGNITION RATE: "+ (double)sum/numberOfParticipants);
    	System.out.println();
    	//System.out.println("TOTAL TRAIN TIME "+ tempEndTrainingTime);
    	System.out.println("TOTAL TEST TIME "+ totalTestTime);
    	
    	System.out.println("AVERAGE TEST TIME "+ totalTestTime/testingImages.length);
    	
    	result[0]=(double)sum/numberOfParticipants;
    	result[1]=totalTestTime/testingImages.length;
    	return result;
    }
    
}//end of class
