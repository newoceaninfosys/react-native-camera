package org.reactnative.camera;

import com.facebook.react.bridge.ReadableArray;

public class ObjectDetectorParams {
    private final String modelFile;
    private final String labelFile;
    private final int inputSize;
    private final int numThreads;
    private final double imageMean;
    private final double imageSTD;
    private final int labelOffset;
    private final boolean isQuantized;
    private final boolean maintainAspect;
    private final double minConfidence;
    private final ReadableArray desiredPreviewSize;

    public ObjectDetectorParams(
            final String modelFile,
            final String labelFile,
            final int inputSize,
            final int numThreads,
            final double imageMean,
            final double imageSTD,
            final int labelOffset,
            final boolean isQuantized,
            final boolean maintainAspect,
            final double minConfidence,
            final ReadableArray desiredPreviewSize) {
        this.modelFile = modelFile;
        this.labelFile = labelFile;
        this.inputSize = inputSize;
        this.numThreads = numThreads;
        this.imageMean = imageMean;
        this.imageSTD = imageSTD;
        this.labelOffset = labelOffset;
        this.isQuantized = isQuantized;
        this.maintainAspect = maintainAspect;
        this.minConfidence = minConfidence;
        this.desiredPreviewSize = desiredPreviewSize;
    }

    public String getModelFile() { return modelFile; }
    public String getLabelFile() { return labelFile; }
    public int getInputSize() { return inputSize; }
    public int getNumThreads() { return numThreads; }
    public double getImageMean() { return imageMean; }
    public double getImageSTD() { return imageSTD; }
    public int getLabelOffset() { return labelOffset; }
    public boolean getIsQuantized() { return isQuantized; }
    public boolean getMaintainAspect() { return maintainAspect; }
    public double getMinConfidence() { return minConfidence; }
    public ReadableArray getDesiredPreviewSize() { return desiredPreviewSize; }
}
