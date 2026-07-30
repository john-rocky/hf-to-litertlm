// LiteRT CompiledModel wrappers for the Bonsai macOS app.
// BonsaiRuntime owns the LiteRtEnvironment; its RuntimeLibraryDir points at
// the app's Frameworks dir so the registry auto-loads the Metal accelerator
// (no explicit registration call needed on macOS, unlike iOS).
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface BonsaiRuntime : NSObject
/// dir must contain libLiteRtMetalAccelerator.dylib (libLiteRt is linked).
- (nullable instancetype)initWithLibraryDir:(NSString *)dir
                                      error:(NSError **)error;
@end

/// One fixed-shape .tflite graph compiled for CPU (XNNPACK) or GPU (Metal,
/// fp32 forced — default fp16 corrupts the Bonsai DiT activations).
@interface BonsaiGraph : NSObject
@property(nonatomic, readonly) double loadSeconds;
@property(nonatomic, readonly) double compileSeconds;
@property(nonatomic, readonly) BOOL fullyAccelerated;
@property(nonatomic, readonly) NSUInteger inputCount;

/// intInputMask: bit k set means host input k (= signature input args_k)
/// is int32; all other inputs and the output are float32.
- (nullable instancetype)initWithRuntime:(BonsaiRuntime *)runtime
                               modelPath:(NSString *)path
                                  useGpu:(BOOL)useGpu
                                 threads:(int)threads
                            intInputMask:(NSUInteger)intInputMask
                                   error:(NSError **)error;

/// inputs[k] feeds signature input args_k; returns output tensor 0 bytes.
- (nullable NSData *)runWithInputs:(NSArray<NSData *> *)inputs
                             error:(NSError **)error
    NS_SWIFT_NAME(run(inputs:));
@end

NS_ASSUME_NONNULL_END
