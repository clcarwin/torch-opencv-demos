Realtime texture_nets
================

This enables realtime stylization with texture networks, for more information check https://github.com/DmitryUlyanov/texture_nets

For now 'starry night' and 'pollock' models are available, we will add more in future.

![screen shot 2016-04-25 at 00 08 15](https://cloud.githubusercontent.com/assets/4953728/14781476/fa8a7c1a-0ae2-11e6-88fb-10e2bf418d86.png)

# Usage

The demo depends on folding Batch Normalization into convolution in [imagine-nn](https://github.com/szagoruyko/imagine-nn), to install it do:

```
luarocks install inn
```

To run on CPU do:

```
OMP_NUM_THREADS=2 th stylization.lua
```

On my dual core macbook it takes about ~0.6s to process one frame.

To run on CUDA do:

```
type=cuda th stylization.lua
```

OpenCL is supported to with `type=cl`. Might be slower than CPU though.

To run on input video file do:

```
video_path=*path to file* th stylization.lua
```

# Model 

[outfile](https://cloud.githubusercontent.com/assets/4953728/14781485/02f31ad8-0ae3-11e6-9cdc-8660c34384b3.png)

# Credits

Thanks to Dmitry Ulyanov for providing the initial version of this demo and the working network.

# Network
```
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> output]
  (1): nn.Concat {
    input
      |`-> (1): nn.Sequential {
      |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
      |      (1): nn.Concat {
      |        input
      |          |`-> (1): nn.Sequential {
      |          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
      |          |      (1): nn.Concat {
      |          |        input
      |          |          |`-> (1): nn.Sequential {
      |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
      |          |          |      (1): nn.Concat {
      |          |          |        input
      |          |          |          |`-> (1): nn.Sequential {
      |          |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
      |          |          |          |      (1): nn.Concat {
      |          |          |          |        input
      |          |          |          |          |`-> (1): nn.Sequential {
      |          |          |          |          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> output]
      |          |          |          |          |      (1): nn.SpatialAveragePooling(32x32, 32,32)
      |          |          |          |          |      (2): nn.NoiseFill
      |          |          |          |          |      (3): nn.Sequential {
      |          |          |          |          |        [input -> (1) -> (2) -> output]
      |          |          |          |          |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |          |          |        (2): nn.SpatialConvolution(3 -> 8, 3x3)
      |          |          |          |          |      }
      |          |          |          |          |      (4): nn.SpatialBatchNormalization
      |          |          |          |          |      (5): nn.LeakyReLU(0.01)
      |          |          |          |          |      (6): nn.Sequential {
      |          |          |          |          |        [input -> (1) -> (2) -> output]
      |          |          |          |          |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |          |          |        (2): nn.SpatialConvolution(8 -> 8, 3x3)
      |          |          |          |          |      }
      |          |          |          |          |      (7): nn.SpatialBatchNormalization
      |          |          |          |          |      (8): nn.LeakyReLU(0.01)
      |          |          |          |          |      (9): nn.SpatialConvolution(8 -> 8, 1x1)
      |          |          |          |          |      (10): nn.SpatialBatchNormalization
      |          |          |          |          |      (11): nn.LeakyReLU(0.01)
      |          |          |          |          |      (12): nn.SpatialUpSamplingNearest
      |          |          |          |          |      (13): nn.SpatialBatchNormalization
      |          |          |          |          |    }
      |          |          |          |          |`-> (2): nn.Sequential {
      |          |          |          |                 [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
      |          |          |          |                 (1): nn.SpatialAveragePooling(16x16, 16,16)
      |          |          |          |                 (2): nn.NoiseFill
      |          |          |          |                 (3): nn.Sequential {
      |          |          |          |                   [input -> (1) -> (2) -> output]
      |          |          |          |                   (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |          |                   (2): nn.SpatialConvolution(3 -> 8, 3x3)
      |          |          |          |                 }
      |          |          |          |                 (4): nn.SpatialBatchNormalization
      |          |          |          |                 (5): nn.LeakyReLU(0.01)
      |          |          |          |                 (6): nn.Sequential {
      |          |          |          |                   [input -> (1) -> (2) -> output]
      |          |          |          |                   (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |          |                   (2): nn.SpatialConvolution(8 -> 8, 3x3)
      |          |          |          |                 }
      |          |          |          |                 (7): nn.SpatialBatchNormalization
      |          |          |          |                 (8): nn.LeakyReLU(0.01)
      |          |          |          |                 (9): nn.SpatialConvolution(8 -> 8, 1x1)
      |          |          |          |                 (10): nn.SpatialBatchNormalization
      |          |          |          |                 (11): nn.LeakyReLU(0.01)
      |          |          |          |                 (12): nn.SpatialBatchNormalization
      |          |          |          |               }
      |          |          |          |           ... -> output
      |          |          |          |      }
      |          |          |          |      (2): nn.Sequential {
      |          |          |          |        [input -> (1) -> (2) -> output]
      |          |          |          |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |          |        (2): nn.SpatialConvolution(16 -> 16, 3x3)
      |          |          |          |      }
      |          |          |          |      (3): nn.SpatialBatchNormalization
      |          |          |          |      (4): nn.LeakyReLU(0.01)
      |          |          |          |      (5): nn.Sequential {
      |          |          |          |        [input -> (1) -> (2) -> output]
      |          |          |          |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |          |        (2): nn.SpatialConvolution(16 -> 16, 3x3)
      |          |          |          |      }
      |          |          |          |      (6): nn.SpatialBatchNormalization
      |          |          |          |      (7): nn.LeakyReLU(0.01)
      |          |          |          |      (8): nn.SpatialConvolution(16 -> 16, 1x1)
      |          |          |          |      (9): nn.SpatialBatchNormalization
      |          |          |          |      (10): nn.LeakyReLU(0.01)
      |          |          |          |      (11): nn.SpatialUpSamplingNearest
      |          |          |          |      (12): nn.SpatialBatchNormalization
      |          |          |          |    }
      |          |          |          |`-> (2): nn.Sequential {
      |          |          |                 [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
      |          |          |                 (1): nn.SpatialAveragePooling(8x8, 8,8)
      |          |          |                 (2): nn.NoiseFill
      |          |          |                 (3): nn.Sequential {
      |          |          |                   [input -> (1) -> (2) -> output]
      |          |          |                   (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |                   (2): nn.SpatialConvolution(3 -> 8, 3x3)
      |          |          |                 }
      |          |          |                 (4): nn.SpatialBatchNormalization
      |          |          |                 (5): nn.LeakyReLU(0.01)
      |          |          |                 (6): nn.Sequential {
      |          |          |                   [input -> (1) -> (2) -> output]
      |          |          |                   (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |                   (2): nn.SpatialConvolution(8 -> 8, 3x3)
      |          |          |                 }
      |          |          |                 (7): nn.SpatialBatchNormalization
      |          |          |                 (8): nn.LeakyReLU(0.01)
      |          |          |                 (9): nn.SpatialConvolution(8 -> 8, 1x1)
      |          |          |                 (10): nn.SpatialBatchNormalization
      |          |          |                 (11): nn.LeakyReLU(0.01)
      |          |          |                 (12): nn.SpatialBatchNormalization
      |          |          |               }
      |          |          |           ... -> output
      |          |          |      }
      |          |          |      (2): nn.Sequential {
      |          |          |        [input -> (1) -> (2) -> output]
      |          |          |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |        (2): nn.SpatialConvolution(24 -> 24, 3x3)
      |          |          |      }
      |          |          |      (3): nn.SpatialBatchNormalization
      |          |          |      (4): nn.LeakyReLU(0.01)
      |          |          |      (5): nn.Sequential {
      |          |          |        [input -> (1) -> (2) -> output]
      |          |          |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |          |        (2): nn.SpatialConvolution(24 -> 24, 3x3)
      |          |          |      }
      |          |          |      (6): nn.SpatialBatchNormalization
      |          |          |      (7): nn.LeakyReLU(0.01)
      |          |          |      (8): nn.SpatialConvolution(24 -> 24, 1x1)
      |          |          |      (9): nn.SpatialBatchNormalization
      |          |          |      (10): nn.LeakyReLU(0.01)
      |          |          |      (11): nn.SpatialUpSamplingNearest
      |          |          |      (12): nn.SpatialBatchNormalization
      |          |          |    }
      |          |          |`-> (2): nn.Sequential {
      |          |                 [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
      |          |                 (1): nn.SpatialAveragePooling(4x4, 4,4)
      |          |                 (2): nn.NoiseFill
      |          |                 (3): nn.Sequential {
      |          |                   [input -> (1) -> (2) -> output]
      |          |                   (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |                   (2): nn.SpatialConvolution(3 -> 8, 3x3)
      |          |                 }
      |          |                 (4): nn.SpatialBatchNormalization
      |          |                 (5): nn.LeakyReLU(0.01)
      |          |                 (6): nn.Sequential {
      |          |                   [input -> (1) -> (2) -> output]
      |          |                   (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |                   (2): nn.SpatialConvolution(8 -> 8, 3x3)
      |          |                 }
      |          |                 (7): nn.SpatialBatchNormalization
      |          |                 (8): nn.LeakyReLU(0.01)
      |          |                 (9): nn.SpatialConvolution(8 -> 8, 1x1)
      |          |                 (10): nn.SpatialBatchNormalization
      |          |                 (11): nn.LeakyReLU(0.01)
      |          |                 (12): nn.SpatialBatchNormalization
      |          |               }
      |          |           ... -> output
      |          |      }
      |          |      (2): nn.Sequential {
      |          |        [input -> (1) -> (2) -> output]
      |          |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |        (2): nn.SpatialConvolution(32 -> 32, 3x3)
      |          |      }
      |          |      (3): nn.SpatialBatchNormalization
      |          |      (4): nn.LeakyReLU(0.01)
      |          |      (5): nn.Sequential {
      |          |        [input -> (1) -> (2) -> output]
      |          |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |          |        (2): nn.SpatialConvolution(32 -> 32, 3x3)
      |          |      }
      |          |      (6): nn.SpatialBatchNormalization
      |          |      (7): nn.LeakyReLU(0.01)
      |          |      (8): nn.SpatialConvolution(32 -> 32, 1x1)
      |          |      (9): nn.SpatialBatchNormalization
      |          |      (10): nn.LeakyReLU(0.01)
      |          |      (11): nn.SpatialUpSamplingNearest
      |          |      (12): nn.SpatialBatchNormalization
      |          |    }
      |          |`-> (2): nn.Sequential {
      |                 [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
      |                 (1): nn.SpatialAveragePooling(2x2, 2,2)
      |                 (2): nn.NoiseFill
      |                 (3): nn.Sequential {
      |                   [input -> (1) -> (2) -> output]
      |                   (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |                   (2): nn.SpatialConvolution(3 -> 8, 3x3)
      |                 }
      |                 (4): nn.SpatialBatchNormalization
      |                 (5): nn.LeakyReLU(0.01)
      |                 (6): nn.Sequential {
      |                   [input -> (1) -> (2) -> output]
      |                   (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |                   (2): nn.SpatialConvolution(8 -> 8, 3x3)
      |                 }
      |                 (7): nn.SpatialBatchNormalization
      |                 (8): nn.LeakyReLU(0.01)
      |                 (9): nn.SpatialConvolution(8 -> 8, 1x1)
      |                 (10): nn.SpatialBatchNormalization
      |                 (11): nn.LeakyReLU(0.01)
      |                 (12): nn.SpatialBatchNormalization
      |               }
      |           ... -> output
      |      }
      |      (2): nn.Sequential {
      |        [input -> (1) -> (2) -> output]
      |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |        (2): nn.SpatialConvolution(40 -> 40, 3x3)
      |      }
      |      (3): nn.SpatialBatchNormalization
      |      (4): nn.LeakyReLU(0.01)
      |      (5): nn.Sequential {
      |        [input -> (1) -> (2) -> output]
      |        (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
      |        (2): nn.SpatialConvolution(40 -> 40, 3x3)
      |      }
      |      (6): nn.SpatialBatchNormalization
      |      (7): nn.LeakyReLU(0.01)
      |      (8): nn.SpatialConvolution(40 -> 40, 1x1)
      |      (9): nn.SpatialBatchNormalization
      |      (10): nn.LeakyReLU(0.01)
      |      (11): nn.SpatialUpSamplingNearest
      |      (12): nn.SpatialBatchNormalization
      |    }
      |`-> (2): nn.Sequential {
             [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
             (1): nn.SpatialAveragePooling(1x1, 1,1)
             (2): nn.NoiseFill
             (3): nn.Sequential {
               [input -> (1) -> (2) -> output]
               (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
               (2): nn.SpatialConvolution(3 -> 8, 3x3)
             }
             (4): nn.SpatialBatchNormalization
             (5): nn.LeakyReLU(0.01)
             (6): nn.Sequential {
               [input -> (1) -> (2) -> output]
               (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
               (2): nn.SpatialConvolution(8 -> 8, 3x3)
             }
             (7): nn.SpatialBatchNormalization
             (8): nn.LeakyReLU(0.01)
             (9): nn.SpatialConvolution(8 -> 8, 1x1)
             (10): nn.SpatialBatchNormalization
             (11): nn.LeakyReLU(0.01)
             (12): nn.SpatialBatchNormalization
           }
       ... -> output
  }
  (2): nn.Sequential {
    [input -> (1) -> (2) -> output]
    (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
    (2): nn.SpatialConvolution(48 -> 48, 3x3)
  }
  (3): nn.SpatialBatchNormalization
  (4): nn.LeakyReLU(0.01)
  (5): nn.Sequential {
    [input -> (1) -> (2) -> output]
    (1): nn.SpatialCircularPadding(l=1,r=1,t=1,b=1)
    (2): nn.SpatialConvolution(48 -> 48, 3x3)
  }
  (6): nn.SpatialBatchNormalization
  (7): nn.LeakyReLU(0.01)
  (8): nn.SpatialConvolution(48 -> 48, 1x1)
  (9): nn.SpatialBatchNormalization
  (10): nn.LeakyReLU(0.01)
  (11): nn.SpatialConvolution(48 -> 3, 1x1)
}
```




