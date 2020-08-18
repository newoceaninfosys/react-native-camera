package org.reactnative.camera.events;

import androidx.core.util.Pools;

import java.util.List;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.uimanager.events.Event;
import com.facebook.react.uimanager.events.RCTEventEmitter;
import org.reactnative.camera.CameraViewManager;
import org.reactnative.camera.tflite.Recognition;
import org.reactnative.camera.utils.ImageDimensions;

public class ObjectDetectedEvent extends Event<ObjectDetectedEvent> {

  private static final Pools.SynchronizedPool<ObjectDetectedEvent> EVENTS_POOL =
          new Pools.SynchronizedPool<>(3);

  private double mScaleX;
  private double mScaleY;
  private List<Recognition> mRecognitions;
  private ImageDimensions mImageDimensions;

  private ObjectDetectedEvent() {}

  public static ObjectDetectedEvent obtain(int viewTag,
                                           List<Recognition> recognitions,
                                           ImageDimensions dimensions,
                                           double scaleX,
                                           double scaleY) {
    ObjectDetectedEvent event = EVENTS_POOL.acquire();
    if (event == null) {
      event = new ObjectDetectedEvent();
    }
    event.init(viewTag, recognitions, dimensions, scaleX, scaleY);
    return event;
  }

  private void init(
          int viewTag,
          List<Recognition> recognitions,
          ImageDimensions dimensions,
          double scaleX,
          double scaleY) {
    super.init(viewTag);
    mRecognitions = recognitions;
    mImageDimensions = dimensions;
    mScaleX = scaleX;
    mScaleY = scaleY;
  }

  @Override
  public String getEventName() {
    return CameraViewManager.Events.EVENT_ON_OBJECT_DETECTED.toString();
  }

  @Override
  public void dispatch(RCTEventEmitter rctEventEmitter) {
    rctEventEmitter.receiveEvent(getViewTag(), getEventName(), serializeEventData());
  }

  private WritableMap serializeEventData() {
    WritableArray dataList = Arguments.createArray();
    for (Recognition reg : mRecognitions) {
      WritableMap event = Arguments.createMap();
      event.putString("label", reg.getTitle());
      event.putDouble("confidence", reg.getConfidence());
      WritableArray locs = Arguments.createArray();
      locs.pushDouble(reg.getLocation().top);
      locs.pushDouble(reg.getLocation().left);
      locs.pushDouble(reg.getLocation().width());
      locs.pushDouble(reg.getLocation().height());
      event.putArray("location", locs);
      dataList.pushMap(event);
    }

    WritableMap event = Arguments.createMap();
    event.putString("type", "objectDetected");
    event.putArray("data", dataList);
    event.putInt("target", getViewTag());

    return event;
  }

//  private WritableMap createEvent() {
//    mData.rewind();
//    byte[] byteArray = new byte[mData.capacity()];
//    mData.get(byteArray);
//    WritableArray dataList = Arguments.createArray();
//    for (byte b : byteArray) {
//      dataList.pushInt((int)b);
//    }
//
//    WritableMap event = Arguments.createMap();
//    event.putString("type", "model");
//    event.putString("prediction", "just test");
//    event.putInt("target", getViewTag());
//
//    return event;
//  }

//  private WritableMap rotateTextX(WritableMap text) {
//    ReadableMap faceBounds = text.getMap("bounds");
//
//    ReadableMap oldOrigin = faceBounds.getMap("origin");
//    WritableMap mirroredOrigin = FaceDetectorUtils.positionMirroredHorizontally(
//        oldOrigin, mImageDimensions.getWidth(), mScaleX);
//
//    double translateX = -faceBounds.getMap("size").getDouble("width");
//    WritableMap translatedMirroredOrigin = FaceDetectorUtils.positionTranslatedHorizontally(mirroredOrigin, translateX);
//
//    WritableMap newBounds = Arguments.createMap();
//    newBounds.merge(faceBounds);
//    newBounds.putMap("origin", translatedMirroredOrigin);
//
//    text.putMap("bounds", newBounds);
//
//    ReadableArray oldComponents = text.getArray("components");
//    WritableArray newComponents = Arguments.createArray();
//    for (int i = 0; i < oldComponents.size(); ++i) {
//      WritableMap component = Arguments.createMap();
//      component.merge(oldComponents.getMap(i));
//      rotateTextX(component);
//      newComponents.pushMap(component);
//    }
//    text.putArray("components", newComponents);
//
//    return text;
//  }

}
