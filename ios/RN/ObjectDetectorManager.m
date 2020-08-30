#import "ObjectDetectorManager.h"
#if __has_include("TFLTensorFlowLite.h")

@interface ObjectDetectorManager ()
//@property(nonatomic, strong) FIRVisionTextRecognizer *textRecognizer;
//@property(nonatomic, assign) float scaleX;
//@property(nonatomic, assign) float scaleY;
@property(nonatomic, strong) TFLInterpreter *interpreter;
@property(nonatomic, strong) TFLTensor *inputTensor;
@property(nonatomic, strong) TFLTensor *resultsOutput;
@property(nonatomic, strong) NSDictionary *options;
@property(nonatomic, strong) NSArray<NSString *> *labels;
@end

@implementation ObjectDetectorManager

- (instancetype)init
{
    if (self = [super init]) {
        NSLog(@"ObjectDetectorManager init");
    }
    return self;
}

-(BOOL)isRealDetector
{
  return true;
}

- (void)load:(NSDictionary *)options
{
    NSLog(@"ObjectDetectorManager load");
    
    self.options = options;
//    NSString* value = [options valueForKey:@"file"];
      
    NSString *modelPath = [NSBundle.mainBundle pathForResource:@"detect"
                                                        ofType:@"tflite"];
    NSError *error = nil;

    TFLInterpreterOptions *tfoptions = [[TFLInterpreterOptions alloc] init];
//    tfoptions.numberOfThreads = 2;
    [tfoptions setNumberOfThreads:2];
    
    self.interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                         options:tfoptions
                                                           error:&error];
    
    if(error != nil || self.interpreter == nil) {
        if(error != nil) {
            NSLog(@"[Error] Could not init Interpreter > %@", [error localizedDescription]);
        } else {
            NSLog(@"[Error] Could not init Interpreter");
        }
        return;
    }
    
    NSString *labelPath = [NSBundle.mainBundle pathForResource:@"labelmap"
                                                        ofType:@"txt"];
    
    NSString *fileContents = [NSString stringWithContentsOfFile:labelPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:&error];
    if(error != nil) {
        NSLog(@"[Error] Could not load labels > %@", [error localizedDescription]);
        return;
    }
    
    self.labels = [fileContents componentsSeparatedByString:@"\n"];
    
    BOOL isQuantized = [[self.options valueForKey:@"isQuantized"] boolValue];
    int inputSize = [[self.options valueForKey:@"inputSize"] integerValue];

    [self.interpreter allocateTensorsWithError:&error];
    if(error != nil) {
        NSLog(@"[Error] allocateTensorsWithError %@", [error localizedDescription]);
        return;
    }
    
    self.inputTensor = [self.interpreter inputTensorAtIndex:0 error:&error];
    if(error != nil) {
        NSLog(@"[Error] inputTensorAtIndex %@", [error localizedDescription]);
        return;
    }
    
    if([self.inputTensor dataType] == TFLTensorDataTypeUInt8 && isQuantized) {
        // QUANTIZED
        NSArray *inputShapes = [self.inputTensor shapeWithError:&error];
        if(error != nil) {
            NSLog(@"[Error] shapeWithError %@", [error localizedDescription]);
            return;
        }
        
        if(inputShapes == nil) {
            NSLog(@"[Error] %@", @"input Tensor's shape is nil!");
            return;
        }
        
        if(([[inputShapes objectAtIndex:0] intValue] != 1) &&
        ([[inputShapes objectAtIndex:1] intValue] != inputSize) &&
        ([[inputShapes objectAtIndex:2] intValue] != inputSize) &&
        ([[inputShapes objectAtIndex:3] intValue] != 3)){
            NSLog(@"[Error] %@", @"input tensor is not correct!");
            return;
        }
        
    } else if([self.inputTensor dataType] == TFLTensorDataTypeFloat32 && !isQuantized){
        // FLoat
    } else {
        // ???
        NSLog(@"[Error] inputTensor not recognized!");
    }
    
}

- (void)run:(UIImage *)uiImage completed: (void (^)(NSArray * result)) completed
{
//    NSLog(@"ObjectDetectorManager run");
    NSMutableArray *recognitionList = [[NSMutableArray alloc] init];
//    completed(data);
    
    int inputSize = [[self.options valueForKey:@"inputSize"] integerValue];
    int labelOffset = [[self.options valueForKey:@"labelOffset"] integerValue];
    float minConfidence = [[self.options valueForKey:@"minConfidence"] integerValue];
    
    NSError *error = nil;
    BOOL result = NO;
    
    NSData *inputData = [self copyDataFromUIImage:uiImage shape:inputSize];
    
    result = [self.inputTensor copyData:inputData error:&error];
    
    if(error != nil) {
        NSLog(@"[Error] run copyData %@", [error localizedDescription]);
        return;
    }
    
    result = [self.interpreter invokeWithError:&error];
    if(error != nil) {
        NSLog(@"[Error] run invokeWithError %@", [error localizedDescription]);
        return;
    }
    
    
    TFLTensor* outputBoundingBox = [self.interpreter outputTensorAtIndex:0 error:&error];
    TFLTensor* outputClasses = [self.interpreter outputTensorAtIndex:1 error:&error];
    TFLTensor* outputScores = [self.interpreter outputTensorAtIndex:2 error:&error];
    TFLTensor* outputCount = [self.interpreter outputTensorAtIndex:3 error:&error];
    
    float dCountValue;
    NSData *outputCountData = [outputCount dataWithError:&error];
    [outputCountData getBytes:&dCountValue length:sizeof(int)];
    int nCount = dCountValue;
    
    float boundingBox[nCount*4];
    NSData *outputBoundingBoxData = [outputBoundingBox dataWithError:&error];
    [outputBoundingBoxData getBytes:&boundingBox length:nCount*4*sizeof(float)];
    
    float classes[nCount];
    NSData *outputClassesData = [outputClasses dataWithError:&error];
    [outputClassesData getBytes:&classes length:nCount*sizeof(float)];
    
    float scores[nCount];
    NSData *outputScoresData = [outputScores dataWithError:&error];
    [outputScoresData getBytes:&scores length:nCount*sizeof(float)];
    
    for (int i=0; i<nCount; i++) {
        if (scores[i] < minConfidence) {
            continue;
        }
        int outputClassIndex = classes[i];
        NSString* outputClass = self.labels[outputClassIndex + labelOffset];
        
//        CGRect rect = CGRectMake(boundingBox[4*i+1],
//                                 boundingBox[4*i],
//                                 boundingBox[4*i+3]-boundingBox[4*i+1],
//                                 boundingBox[4*i+2]-boundingBox[4*i]);
        
        [recognitionList addObject:@{@"confidence": [NSNumber numberWithFloat:scores[i]], @"label": outputClass}];
    }
    
    completed(recognitionList);
    
    
//    TFLTensor *outputTensor = [self.interpreter outputTensorAtIndex:0 error:&error];
//    // read results from output tensor
//    NSData *outputData = [outputTensor dataWithError:&error];
//    // it might not be outputData long, but it can't be longer than outputData
//    NSMutableArray<NSDictionary *> *results = [NSMutableArray arrayWithCapacity:[outputData length]];
//
//    float threshold = minConfidence;
//
//    // while it would normally be preferrable to enumerate the bytes so they don't get flattened
//    // that doesn't matter in this case
//    uint8_t *outputBytes = (uint8_t *)[outputData bytes];
//    for (uint i = 0; i < [outputData length]; ++i) {
//        float confidence = (float)((outputBytes[i] & 0xff) / 255.0f);
//        if (confidence >= threshold) {
//            // get label and add to array
//            NSString *label = i < [self.labels count] ? self.labels[i] : @"unknown";
//            [results addObject:@{@"confidence": [NSNumber numberWithFloat:confidence], @"label": label}];
//        }
//    }
//
//    // sort descending and return only the min(count, numResults)
//    recognitionList = [results sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
//        NSNumber *first = [(NSDictionary *)a objectForKey:@"confidence"];
//        NSNumber *second = [(NSDictionary *)b objectForKey:@"confidence"];
//        return [second compare:first];
//    }];
//
//    int numResults = 5;
//
//    if ([recognitionList count] <= numResults) {
//        completed(recognitionList);
//    } else {
//        NSRange subset;
//        subset.location = 0;
//        subset.length = numResults;
//        completed([recognitionList subarrayWithRange:subset]);
//    }
}

-(UIImage *) cropImage:(UIImage *) uiImage {
    double shape = uiImage.size.width;
    CGRect cropRect = CGRectMake(0, 0, shape, shape);
    CGImageRef imageRef = CGImageCreateWithImageInRect([uiImage CGImage], cropRect);
    UIImage *cropped = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    return cropped;
}

-(UIImage *) resizeImage:(UIImage *) uiImage
                   shape:(int)shape
{
    CGSize size = CGSizeMake(shape, shape);
    UIGraphicsBeginImageContext(size);
    [uiImage drawInRect:CGRectMake(0, 0, size.width, size.height)];
    UIImage *destImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return destImage;
}

-(NSData *) copyDataFromUIImage:(UIImage *) uiImage shape:(int)shape {
    // Crop
    UIImage *newImage = [self cropImage:uiImage];
    
    // Resize
    newImage = [self resizeImage:newImage shape:shape];
    
    CGImageRef image = newImage.CGImage;
    long imageWidth = CGImageGetWidth(image);
    long imageHeight = CGImageGetHeight(image);
    CGContextRef context = CGBitmapContextCreate(nil,
                                                 imageWidth, imageHeight,
                                                 8,
                                                 imageWidth * 4,
                                                 CGColorSpaceCreateDeviceRGB(),
                                                 kCGImageAlphaNoneSkipFirst);
    CGContextDrawImage(context, CGRectMake(0, 0, imageWidth, imageHeight), image);
    
    NSMutableData *inputData = [NSMutableData dataWithLength:(1 * shape * shape * 3)];
    uint8_t *scaledRgbPixels = [inputData mutableBytes];
    uint8_t *scaledArgbPixels = (uint8_t *)CGBitmapContextGetData(context);

    uint scaledRgbOffset = 0;
    uint scalledArgbOffset = 1;
    for (uint y = 0; y < shape; ++y) {
        for (uint x = 0; x < shape; ++x, scalledArgbOffset++) {
            scaledRgbPixels[scaledRgbOffset++] = scaledArgbPixels[scalledArgbOffset++];
            scaledRgbPixels[scaledRgbOffset++] = scaledArgbPixels[scalledArgbOffset++];
            scaledRgbPixels[scaledRgbOffset++] = scaledArgbPixels[scalledArgbOffset++];
        }
    }

    CGContextRelease(context);

    
//    UInt8 *imageData = CGBitmapContextGetData(context);
//
//    NSMutableData *inputData = [[NSMutableData alloc] initWithCapacity:0];
//
//    for (int row = 0; row < shape; row++) {
//      for (int col = 0; col < shape; col++) {
//        long offset = 3 * (col * imageWidth + row);
//        // Normalize channel values to [0.0, 1.0]. This requirement varies
//        // by model. For example, some models might require values to be
//        // normalized to the range [-1.0, 1.0] instead, and others might
//        // require fixed-point values or the original bytes.
//        // (Ignore offset 0, the unused alpha channel)
////        Float32 red = imageData[offset+1] / 255.0f;
////        Float32 green = imageData[offset+2] / 255.0f;
////        Float32 blue = imageData[offset+3] / 255.0f;
//
//        uint8_t red = imageData[offset+1] / 255.0f;
//        uint8_t green = imageData[offset+2] / 255.0f;
//        uint8_t blue = imageData[offset+3] / 255.0f;
//
//        [inputData appendBytes:&red length:sizeof(red)];
//        [inputData appendBytes:&green length:sizeof(green)];
//        [inputData appendBytes:&blue length:sizeof(blue)];
//      }
//    }
    
    return inputData;
}

@end
#else

@interface ObjectDetectorManager ()
@end

@implementation ObjectDetectorManager

- (instancetype)init
{
  self = [super init];
  return self;
}

-(BOOL)isRealDetector
{
  return false;
}

-(void)run:(UIImage *)image completed:(postRecognitionBlock)completed;
{
  NSLog(@"ObjectDetector not installed, stub used!");
  NSArray *features = @[@"Error, Object Detector not installed"];
  completed(features);
}

-(void)load:(NSDictionary *)options
{
  NSLog(@"ObjectDetector not installed, stub used!");
}

@end
#endif
