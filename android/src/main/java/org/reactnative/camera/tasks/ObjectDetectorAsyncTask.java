package org.reactnative.camera.tasks;

import org.reactnative.camera.ObjectDetectorParams;
import org.reactnative.camera.Recognition;
import org.tensorflow.lite.Interpreter;

import android.graphics.Bitmap;
import android.graphics.RectF;

import java.nio.ByteBuffer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public class ObjectDetectorAsyncTask extends android.os.AsyncTask<Void, Void, List<Recognition>> {

  private ObjectDetectorAsyncTaskDelegate mDelegate;
  private Interpreter mModelProcessor;
  private ByteBuffer mInputBuf;
  private Vector<String> mModelLabels;
  private int mWidth;
  private int mHeight;
  private int mRotation;
  private Bitmap mCroppedBitmap;
  private ObjectDetectorParams mParams;

  private static final int NUM_DETECTIONS = 10;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  public ObjectDetectorAsyncTask(
      ObjectDetectorAsyncTaskDelegate delegate,
      Interpreter modelProcessor,
      Vector<String> labels,
      ByteBuffer inputBuf,
      int width,
      int height,
      int rotation,
      Bitmap croppedBitmap,
      ObjectDetectorParams params
  ) {
    mDelegate = delegate;
    mModelProcessor = modelProcessor;
    mModelLabels = labels;
    mInputBuf = inputBuf;
    mWidth = width;
    mHeight = height;
    mRotation = rotation;
    mCroppedBitmap = croppedBitmap;
    mParams = params;
  }
    
  @Override
  protected List<Recognition> doInBackground(Void... ignored) {
    if (isCancelled() || mDelegate == null || mModelProcessor == null) {
      return null;
    }

    mInputBuf.rewind();

    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];

    Object[] inputArray = {mInputBuf};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);

    mModelProcessor.runForMultipleInputsOutputs(inputArray, outputMap);

    final ArrayList<Recognition> recognitions = new ArrayList<>(5);

    // Show the best detections.
    // after scaling them back to the input size.

    // You need to use the number of detections from the output and not the NUM_DETECTONS variable declared on top
    // because on some models, they don't always output the same total number of detections
    // For example, your model's NUM_DETECTIONS = 20, but sometimes it only outputs 16 predictions
    // If you don't use the output's numDetections, you'll get nonsensical data
    int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) numDetections[0]); // cast from float to integer, use min for safety

    for (int i = 0; i < numDetectionsOutput; ++i) {
      final RectF detection =
              new RectF(
                      outputLocations[0][i][1] * mParams.getInputSize(),
                      outputLocations[0][i][0] * mParams.getInputSize(),
                      outputLocations[0][i][3] * mParams.getInputSize(),
                      outputLocations[0][i][2] * mParams.getInputSize());

      // SSD Mobilenet V1 Model assumes class 0 is background class
      // in label file and class labels start from 1 to number_of_classes+1,
      // while outputClasses correspond to class index from 0 to number_of_classes
      int labelOffset = mParams.getLabelOffset();
      recognitions.add(
              new Recognition(
                      "" + i,
                      mModelLabels.get((int) outputClasses[0][i] + labelOffset),
                      outputScores[0][i],
                      detection));
    }

    return recognitions;
  }

  @Override
  protected void onPostExecute(List<Recognition> recognitions) {
    super.onPostExecute(recognitions);

    if (recognitions != null) {
      mDelegate.onObjectDetected(recognitions, mWidth, mHeight, mRotation, mCroppedBitmap);
    }
    mDelegate.onObjectDetectorTaskCompleted();
  }
}
