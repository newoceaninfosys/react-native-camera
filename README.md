# React Native Camera with Tensorflow Object Detector
This fork is about to add object dection using tensorflow tflite.
Inspired by:
- `https://github.com/shaqian/tflite-react-native`
- `https://github.com/Macilias/react-native-camera-tflite-updated`

## Docs
Follow main docs of React native Camera at here [https://react-native-community.github.io/react-native-camera/](https://react-native-community.github.io/react-native-camera/)

#### How to use this fork?

**yarn**: `yarn add react-native-camera@git+https://git@github.com/newoceaninfosys/react-native-camera.git`

**npm**: `npm install --save react-native-camera@git+https://git@github.com/newoceaninfosys/react-native-camera.git`

### Extra Installations:
#### Android
- Add following to `app/build.gradle`
```
android {
    ...
    aaptOptions {
        noCompress 'tflite'
    }
    ...
}
```

### Example usage (Android only, iOS comming soon)

```jsx
import React, {useState, useRef, useCallback} from 'react';
import {StyleSheet, View, Text} from 'react-native';
import _ from 'lodash';
import { RNCamera } from 'react-native-camera';

const App: () => React$Node = () => {
  const cameraRef = useRef();
  const [prediction, setPrediction] = useState('N/A');
  const processOutput = ({data}) => {
    const output = _.take(_.orderBy(_.filter(data, r => r.confidence > 0.5), 'confidence'), 2)
      .map((r) => r.label + ' - ' + r.confidence)
      .join('\n');
    setPrediction(output);
  };

  // Download TFLite model at here then place in android/app/src/assets
  // https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
  const modelParams = {
    file: 'detect.tflite',
    label: 'labelmap.txt'
  };

  const onCameraReady = useCallback(() => {
    console.log('Load Model', modelParams);
    cameraRef.current.loadObjectDetectorModel(modelParams);
  }, [modelParams]);

  return (
    <View style={styles.container}>
      <RNCamera
        ref={cameraRef}
        style={styles.preview}
        type={RNCamera.Constants.Type.back}
        flashMode={RNCamera.Constants.FlashMode.on}
        onCameraReady={onCameraReady}
        androidCameraPermissionOptions={{
          title: 'Permission to use camera',
          message: 'We need your permission to use your camera',
          buttonPositive: 'Ok',
          buttonNegative: 'Cancel',
        }}
        androidRecordAudioPermissionOptions={{
          title: 'Permission to use audio recording',
          message: 'We need your permission to use your audio',
          buttonPositive: 'Ok',
          buttonNegative: 'Cancel',
        }}
        onObjectDetected={processOutput}
      />
      <View style={{flex: 0, flexDirection: 'row', justifyContent: 'center', height: 100, alignItems: 'center'}}>
        <Text>{prediction}</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'column',
    backgroundColor: 'white',
  },
  preview: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
});

export default App;
```

![IbWBAaD](https://i.imgur.com/IbWBAaD.gif)