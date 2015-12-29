package imageAnalysisPackage;

import static org.bytedeco.javacpp.opencv_contrib.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;

import java.awt.List;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Random;

import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_contrib.FaceRecognizer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.opencv.core.Size;
///WRONG APPROACH!!!!!
public class test_LBPH_Time_updates {
	
	static String datasetLabels [] = {"WITH_LIGHT"};//,"WITH_LIGHT"};//{"WITH_LIGHT"};//,"WITHOUT_LIGHT"};
	
	final static Size outImageSize = new Size(500,750);
	final static boolean equalizeHistograms = false;
	static int NUM_OF_RUNS=1;
	static double distTreshold = 100; //distance threshold   
	static FaceRecognizer faceRecognizer =createLBPHFaceRecognizer(9, 9, 9, 9, distTreshold);
	
	
	public static void main(String args[]) throws IOException{
		
		
		for(String datasetLabel: datasetLabels){
			String trainingDir = "./DATASETS_COMBINED/train_"+datasetLabel;//args[0];
			String testingDir="./DATASETS_COMBINED/test_"+datasetLabel;
			File trainDir=new File(trainingDir);
			File[] trainingImages = trainDir.listFiles();
			
			File testDir=new File(testingDir);
			File[] testingImages = testDir.listFiles();
			
			System.out.println("NUMBER OF TRAINING IMAGES "+trainingImages.length);
			System.out.println("NUMBER OF TEST IMAGES "+testingImages.length);
			
			int NUM_OF_PARTICIPANTS=trainingImages.length;
			
			
			double[] avgUpdateTime = new double[NUM_OF_PARTICIPANTS];
			double[] avgRecognitionRate = new double[NUM_OF_PARTICIPANTS];
			double[] avgTestTime = new double[NUM_OF_PARTICIPANTS];
			
			
			//for(int run=1; run<=NUM_OF_RUNS;run++){
				MatVector images = new MatVector(trainingImages.length);
				Mat labels = new Mat(trainingImages.length, 1, CV_32SC1);
				IntBuffer labelsBuf = labels.getIntBuffer();
				
				int counter = 0;
				
				
				//form training set
			    for (File image : trainingImages) {
//			          Mat img = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC1);
			          Mat img = new Mat((int)outImageSize.height, (int)outImageSize.width,org.bytedeco.javacpp.opencv_core.CV_8UC3);
			          
			          System.out.println(image.getPath());
			          //LBPH only works with GRAYSCALE images

			          img = org.bytedeco.javacpp.opencv_highgui.imread(image.getAbsolutePath(), org.bytedeco.javacpp.opencv_highgui.CV_LOAD_IMAGE_GRAYSCALE);
			          org.bytedeco.javacpp.opencv_imgproc.resize(img, img, new org.bytedeco.javacpp.opencv_core.Size(500,750));
			          if(equalizeHistograms)
			        	  opencv_imgproc.equalizeHist(img, img);

			          int label = Integer.parseInt(image.getName().split("\\_")[0]);
			          
			          //System.out.println(image.getName());
			          
			          images.put(counter, img);
			
			          labelsBuf.put(counter, label);
			
			          counter++;
			      }//end of forming training set
			    
			  
			    
			    
			     ArrayList<Integer> usedLabelIndices = new ArrayList<Integer>();
			     Random random = new Random();
				    
			    //randomly generating first participant
			    int firstParticipant = random.nextInt(trainingImages.length/2) * 2;
			    usedLabelIndices.add(firstParticipant);
			    
			    System.out.println("first participant's index  " +firstParticipant );
			    System.out.println(labelsBuf.get(firstParticipant));
			    System.out.println(labelsBuf.get(firstParticipant+1));
				

			    //randomly generating second participant
			    int secondParticipant = random.nextInt(trainingImages.length/2) * 2;
			    
			    while(usedLabelIndices.contains(secondParticipant)){
			    	if(usedLabelIndices.size()!=trainingImages.length/2)
			    		secondParticipant = random.nextInt(trainingImages.length/2) * 2;
			    	else
			    		break;
			    }
			    
			    
			    System.out.println("second participant's index  " +secondParticipant );
			    System.out.println(labelsBuf.get(secondParticipant));
			    System.out.println(labelsBuf.get(secondParticipant+1));
			    
			    //currently there are only 2 participants in our system 
			   
			    
			    MatVector initImages = new MatVector(4);
				Mat initLabels = new Mat(4, 1, CV_32SC1);
				IntBuffer initLabelsBuf = initLabels.getIntBuffer();
			    
				initImages.put(0,images.get(firstParticipant));
				initImages.put(1,images.get(firstParticipant+1));
				initImages.put(2,images.get(secondParticipant));
				initImages.put(3,images.get(secondParticipant+1));
				
				
				initLabelsBuf.put(0, labelsBuf.get(firstParticipant));
				initLabelsBuf.put(1, labelsBuf.get(firstParticipant+1));
				initLabelsBuf.put(2, labelsBuf.get(secondParticipant));
				initLabelsBuf.put(3, labelsBuf.get(secondParticipant+1));
			    
//				ProcessImages.displayImage(images.get(firstParticipant).getBufferedImage(), "first participant label "+labelsBuf.get(firstParticipant));
//				ProcessImages.displayImage(images.get(firstParticipant+1).getBufferedImage(), "first participant label "+labelsBuf.get(firstParticipant+1));
//				ProcessImages.displayImage(images.get(secondParticipant).getBufferedImage(), "second participant label "+labelsBuf.get(secondParticipant));
//				ProcessImages.displayImage(images.get(secondParticipant+1).getBufferedImage(), "second participant label "+labelsBuf.get(secondParticipant+1));
				
				
				
					
				//now create FaceRecognizer with these 2 labels
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////
				

			    //long processingTime = System.currentTimeMillis()-startTime; 
			    faceRecognizer.set("threshold", 7500.0);//too big numbers
			    long tempStartTrainTime = System.currentTimeMillis();
				//test with default facerecognizer
			    //FaceRecognizer faceRecognizer =createLBPHFaceRecognizer();
			    faceRecognizer.train(initImages, initLabels);
			    long tempEndTrainTime = System.currentTimeMillis()-tempStartTrainTime;
			    initImages.get(0).release();
			    initImages.get(1).release();
			    initImages.get(2).release();
			    initImages.get(3).release();
			    initLabels.release();
			    initImages.deallocate();
			    initLabels.deallocate();
			    initLabelsBuf.clear();
			    
			    System.gc();
			    
			    ///////////////////////////////////////////////////////////////////////////////////////
			    ///////////////////////////////////////////////////////////////////////////////////////
			    ///////////////////////////////////////////////////////////////////////////////////////
			    
			  //now test the model on the dataset
			    
				double []res = testOnDataset(testingImages, 2);
				
				avgUpdateTime[0]+=tempEndTrainTime;
				avgRecognitionRate[0]+=res[0];
				avgTestTime[0]+=res[1];
				//then start adding participants:
				//TODO:
				
				for( int numOfParticipants=2;numOfParticipants<NUM_OF_PARTICIPANTS;numOfParticipants++){
					System.out.println();
					System.out.println("NOW THERE ARE "+numOfParticipants + " PARTICIPANTS IN SYSTEM");
					//randomly generate current participant
					// add the third participant 
				    
					MatVector  tmpImages = new MatVector(2);
					Mat tmpLabels = new Mat(2, 1, CV_32SC1);
					IntBuffer tmpLabelsBuf = tmpLabels.getIntBuffer();
					
					
					//randomly generating second participant
				    int currentParticipant = random.nextInt(trainingImages.length/2) * 2;
				    
				    while(usedLabelIndices.contains(currentParticipant)){
				    	if(usedLabelIndices.size()!=trainingImages.length/2)
				    		currentParticipant = random.nextInt(trainingImages.length/2) * 2;
				    	else
				    		break;
				    } 
					
				    System.out.println("current participant's index  " +currentParticipant );
				    System.out.println(labelsBuf.get(currentParticipant));
				    System.out.println(labelsBuf.get(currentParticipant+1));
				    
				    tmpImages.put(0,images.get(currentParticipant));
				    tmpImages.put(1,images.get(currentParticipant+1));
					tmpLabelsBuf.put(0, labelsBuf.get(currentParticipant));
					tmpLabelsBuf.put(1, labelsBuf.get(currentParticipant+1));
					
//					ProcessImages.displayImage(images.get(currentParticipant).getBufferedImage(), "third participant label "+labelsBuf.get(thirdParticipant));
//					ProcessImages.displayImage(images.get(currentParticipant+1).getBufferedImage(), "third participant label "+labelsBuf.get(thirdParticipant+1));
					
					//update the face recognizer
					long tempStartUpdateTime = System.currentTimeMillis();
					faceRecognizer.update(tmpImages, tmpLabels);
					long tempEndUpdateTime =System.currentTimeMillis()- tempStartUpdateTime; 
					
					
					tmpImages.get(0).release();
					tmpImages.get(1).release();
					tmpImages.deallocate();
					tmpLabels.deallocate();
					tmpLabelsBuf.clear();
					System.gc();
					
					
					System.out.println("UPDATED IN "+ tempEndUpdateTime + "  msecs");
					
					
					
					
					//now test the model on the dataset
					res=testOnDataset( testingImages, numOfParticipants+1);
					
					
					/////////////////////////////////////////////////////////////////////////////////////////
					//predicting
					////////////////////////////////////////////////////////////////////////////////////////
					

					////////////////////////////////////////////////////////////////////////////////////
					////////////////////////////////////////////////////////////////////////////////////
					
					avgUpdateTime[numOfParticipants-1]+=tempEndUpdateTime;
					avgRecognitionRate[numOfParticipants-1]+=res[0];
					avgTestTime[numOfParticipants-1]+=res[1];
					
					
					
					
				}//end of for iterating over all test images
			//}//end of iterating 10 times
			
			
			
			
			//write to file
			FileWriter writer = new FileWriter("results_LBPH_Updating"+datasetLabel+".csv ");
			
			writer.append("number of participants");
		    writer.append(',');
		    writer.append("recognition rate");
		    writer.append(',');
		    writer.append("training/update time");
		    writer.append(',');
		    writer.append("testing/recall time");
		    writer.append('\n');
			
			
			for(int i=0; i<NUM_OF_PARTICIPANTS; i++){
				System.out.println((i+2)+"  "+(double)avgRecognitionRate[i]/NUM_OF_RUNS+"  "+(double)avgUpdateTime[i]/NUM_OF_RUNS+"  "+(double)avgTestTime[i]/NUM_OF_RUNS);
				//SAVE TO FILE
		    	writer.append(String.valueOf(i+2));
			    writer.append(',');
			    writer.append(String.valueOf((double)avgRecognitionRate[i]/NUM_OF_RUNS));
			    writer.append(',');
			    writer.append(String.valueOf((double)avgUpdateTime[i]/NUM_OF_RUNS));
			    writer.append(',');
			    writer.append(String.valueOf((double)avgTestTime[i]/NUM_OF_RUNS));
			    writer.append('\n');
			}//end of for
			
			System.out.println("DONE PROCESING "+datasetLabel+".......................................................................");
			writer.flush();
			writer.close();
		}//end iterating over two categories of data
		
	}//end of main
/**
 * 	
 * @param testingImages
 * @param faceRecognizer
 * @param numberOfParticipants
 */
public static double[] testOnDataset(File[] testingImages,int numberOfParticipants){
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
	
		if(equalizeHistograms)
			opencv_imgproc.equalizeHist(testImage, testImage);
	
	
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
	
//		System.out.println("Recall Time for current image: " + tempEndTestTime);
//	
//		System.out.println();
//		System.out.println("Processing image "+image.getName());
//		System.out.println("-----------------------------------------------------------------------------------");
//	
//		System.out.println("Predicted label: " + predictedLabel[0]);
//		System.out.println("Actual label: " + actualLabel);
//		System.out.println("Prediction Confidence: " + confidence[0]);
	
	
		i++;
		}
		System.out.println();
		
		//faceRecognizer.deallocate();
		//System.gc();
		
		
		System.out.println("Correctly recognized "+sum +" out of "+numberOfParticipants);
		System.out.println("RECOGNITION RATE: "+ (double)sum/numberOfParticipants);
		System.out.println();
		//System.out.println("TOTAL TRAIN TIME "+ tempEndTrainingTime);
		System.out.println("TOTAL TEST TIME "+ totalTestTime);
		
		System.out.println("AVERAGE TEST TIME "+ totalTestTime/testingImages.length);
		

		//for testing when all the participants are in the system
//		System.out.println("Correctly recognized "+sum +" out of "+testingImages.length);
//		System.out.println("RECOGNITION RATE: "+ (double)sum/testingImages.length);
//		System.out.println();
//		//System.out.println("TOTAL TRAIN TIME "+ tempEndTrainingTime);
//		System.out.println("TOTAL TEST TIME "+ totalTestTime);
//		
//		System.out.println("AVERAGE TEST TIME "+ totalTestTime/testingImages.length);
		
		
		result[0]=(double)sum/numberOfParticipants;
		result[1]=totalTestTime/testingImages.length;
		return result;
	}


}//end of class





