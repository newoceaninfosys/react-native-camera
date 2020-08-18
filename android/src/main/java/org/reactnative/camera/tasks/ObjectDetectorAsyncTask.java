package org.reactnative.camera.tasks;

import org.reactnative.camera.tflite.ObjectDetectionAPI;
import org.reactnative.camera.tflite.ObjectDetectorParams;
import org.reactnative.camera.tflite.Recognition;
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
  private int mWidth;
  private int mHeight;
  private int mRotation;
  private Bitmap mCroppedBitmap;
  private ObjectDetectionAPI mObjectDetectorAPI;

  public ObjectDetectorAsyncTask(
      ObjectDetectorAsyncTaskDelegate delegate,
      ObjectDetectionAPI objectDetectorAPI,
      int width,
      int height,
      int rotation,
      Bitmap croppedBitmap
  ) {
    mDelegate = delegate;
    mObjectDetectorAPI = objectDetectorAPI;
    mWidth = width;
    mHeight = height;
    mRotation = rotation;
    mCroppedBitmap = croppedBitmap;
  }
    
  @Override
  protected List<Recognition> doInBackground(Void... ignored) {
    if (isCancelled() || mDelegate == null || mObjectDetectorAPI == null || mObjectDetectorAPI.getDetector() == null) {
      return null;
    }

    return mObjectDetectorAPI.run(mCroppedBitmap);
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
