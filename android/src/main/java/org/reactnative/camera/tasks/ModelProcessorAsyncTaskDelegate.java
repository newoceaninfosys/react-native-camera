package org.reactnative.camera.tasks;

import android.graphics.Bitmap;

import org.reactnative.camera.Recognition;

import java.nio.ByteBuffer;
import java.util.List;

public interface ModelProcessorAsyncTaskDelegate {
  void onModelProcessed(List<Recognition> recognitions, int sourceWidth, int sourceHeight, int sourceRotation, Bitmap croppedBitmap);
  void onModelProcessorTaskCompleted();
}
