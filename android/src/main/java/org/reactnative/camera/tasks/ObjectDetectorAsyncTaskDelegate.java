package org.reactnative.camera.tasks;

import android.graphics.Bitmap;

import org.reactnative.camera.tflite.Recognition;

import java.util.List;

public interface ObjectDetectorAsyncTaskDelegate {
  void onObjectDetected(List<Recognition> recognitions, int sourceWidth, int sourceHeight, int sourceRotation, Bitmap croppedBitmap);
  void onObjectDetectorTaskCompleted();
}
