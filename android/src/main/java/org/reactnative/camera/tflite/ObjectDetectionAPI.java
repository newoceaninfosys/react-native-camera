package org.reactnative.camera.tflite;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Environment;

import androidx.core.app.ActivityCompat;

import com.facebook.react.uimanager.ThemedReactContext;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public class ObjectDetectionAPI {
    private Vector<String> labels = new Vector<String>();
    private static final int NUM_DETECTIONS = 10;
    private Interpreter _objectDetector;
    private ThemedReactContext _context;
    private ObjectDetectorParams _params;
    private ByteBuffer _imgData;

    private int[] _intValues;
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private float[][][] _outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] _outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private float[][] _outputScores;
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private float[] _numDetections;

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    public ObjectDetectionAPI(ThemedReactContext context, ObjectDetectorParams params) {
        this._context = context;
        this._params = params;
        try {
            InputStream labelInputStream = context.getAssets().open(params.getLabelFile());
            BufferedReader br = new BufferedReader(new InputStreamReader(labelInputStream));
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            br.close();

            _objectDetector = new Interpreter(loadModelFile());
            _objectDetector.setNumThreads(params.getNumThreads());

            int numBytesPerChannel;
            if (_params.getIsQuantized()) {
                numBytesPerChannel = 1; // Quantized
            } else {
                numBytesPerChannel = 4; // Floating point
            }

            int inputSize = params.getInputSize();

//            ByteBuffer.allocateDirect(_params.getInputSize() * _params.getInputSize() * 3);
            _imgData = ByteBuffer.allocateDirect(1 *  inputSize * inputSize * 3 * numBytesPerChannel);
            _imgData.order(ByteOrder.nativeOrder());
            _intValues = new int[inputSize * inputSize];

            _outputLocations = new float[1][NUM_DETECTIONS][4];
            _outputClasses = new float[1][NUM_DETECTIONS];
            _outputScores = new float[1][NUM_DETECTIONS];
            _numDetections = new float[1];
        } catch(Exception e) {
            e.printStackTrace();
            _objectDetector = null;
        }
    }

    public void saveImage(Bitmap img, String fn) {
        try {
            int permission = ActivityCompat.checkSelfPermission(_context.getCurrentActivity(), Manifest.permission.WRITE_EXTERNAL_STORAGE);

            if (permission != PackageManager.PERMISSION_GRANTED) {
                // We don't have permission so prompt the user
                ActivityCompat.requestPermissions(
                        _context.getCurrentActivity(),
                        PERMISSIONS_STORAGE,
                        REQUEST_EXTERNAL_STORAGE
                );
            } else {
                String file_path = Environment.getExternalStorageDirectory().getAbsolutePath() + "/tensorflow";
                File dir = new File(file_path);
                if(!dir.exists())
                    dir.mkdirs();
                File file = new File(dir, fn);
                if(file.exists())
                    file.delete();
                FileOutputStream fOut = new FileOutputStream(file, false);

                img.compress(Bitmap.CompressFormat.PNG, 85, fOut);
                fOut.flush();
                fOut.close();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public List<Recognition> run(Bitmap croppedBitmap) {
        if(_params.getSavePreview())
            saveImage(croppedBitmap, "preview.png");

        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        croppedBitmap.getPixels(_intValues, 0, croppedBitmap.getWidth(), 0, 0, croppedBitmap.getWidth(), croppedBitmap.getHeight());

        _imgData.rewind();

        float imageMean = (float)_params.getImageMean();
        float imageSTD = (float)_params.getImageSTD();
        for (int i = 0; i < _params.getInputSize(); ++i) {
            for (int j = 0; j < _params.getInputSize(); ++j) {
                final int pixelValue = _intValues[i * _params.getInputSize() + j];
                if(_params.getIsQuantized()) {
                    // Quantized model
                    _imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    _imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    _imgData.put((byte) (pixelValue & 0xFF));
                } else {
                    // Float model
                    _imgData.putFloat((((pixelValue >> 16) & 0xFF) - imageMean) / imageSTD);
                    _imgData.putFloat((((pixelValue >> 8) & 0xFF) - imageMean) / imageSTD);
                    _imgData.putFloat(((pixelValue & 0xFF) - imageMean) / imageSTD);
                }
            }
        }

        // Copy the input data into TensorFlow.
        _outputLocations = new float[1][NUM_DETECTIONS][4];
        _outputClasses = new float[1][NUM_DETECTIONS];
        _outputScores = new float[1][NUM_DETECTIONS];
        _numDetections = new float[1];

        Object[] inputArray = {_imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, _outputLocations);
        outputMap.put(1, _outputClasses);
        outputMap.put(2, _outputScores);
        outputMap.put(3, _numDetections);

        // Run the inference call.
        _objectDetector.runForMultipleInputsOutputs(inputArray, outputMap);

        final ArrayList<Recognition> recognitions = new ArrayList<>(5);

        // Show the best detections.
        // after scaling them back to the input size.

        // You need to use the number of detections from the output and not the NUM_DETECTONS variable declared on top
        // because on some models, they don't always output the same total number of detections
        // For example, your model's NUM_DETECTIONS = 20, but sometimes it only outputs 16 predictions
        // If you don't use the output's numDetections, you'll get nonsensical data
        int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) _numDetections[0]); // cast from float to integer, use min for safety

        for (int i = 0; i < numDetectionsOutput; ++i) {
            final RectF detection =
                    new RectF(
                            _outputLocations[0][i][1] * _params.getInputSize(),
                            _outputLocations[0][i][0] * _params.getInputSize(),
                            _outputLocations[0][i][3] * _params.getInputSize(),
                            _outputLocations[0][i][2] * _params.getInputSize());

            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes+1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            int labelOffset = _params.getLabelOffset();
            recognitions.add(
                    new Recognition(
                            "" + i,
                            labels.get((int) _outputClasses[0][i] + labelOffset),
                            _outputScores[0][i],
                            detection));
        }

        return recognitions;
    }

    public void destroy() {
        if(_objectDetector != null)
            _objectDetector.close();
        _objectDetector = null;
    }

    public Vector<String> getLabels() { return labels; }

    public ObjectDetectorParams getParams() { return _params; }

    public Interpreter getDetector() { return _objectDetector; }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = _context.getAssets().openFd(_params.getModelFile());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
