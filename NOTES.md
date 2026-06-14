
## Training Setup

A cluster of two Quadro RTX 6000 GPUs

## Architecture

## Losses

### Gram Loss

### HSIC Loss

### Centering vs Sinkhorn-Knopp

You should only use SinkhornKnopp loss if your batch size is large enough to allow for dense
clustering per prototype.

For DINOv3, Meta uses 131,072 prototypes and a batch size of 4096 (256 per GPU). This means
that for every loss calculation, there are 4096 * 2 * (224/16)**2 = 1,605,632 global feature
datapoints and 4096 * 8 * (96/16)**2 = 1,179,648 local feature datapoints.

The prototype density for global features is 1,605,632 / 131,072 = 12.25 and for local features
it is 1,179,648 / 131,072 = 9.0.

Given that we are using ConvNeXt-v2 the math changes a bit (we downsample by 32 instead of 16)
but the idea is the same. For 4,096 prototypes and a batch size of 60 (30 per GPU), global
feature datapoint count is 60 * 2 * (224/32)**2 = 5,880 and local feature datapoint count is
60 * 8 * (96/32)**2 = 4,320. The prototype density becomes 5,880 / 4,096 = 1.44 for global features
and 4,320 / 4,096 = 1.05 for local features. Almost 9 times less dense than Meta's config.

As such, in our case due to the limited resources present, we have to use DINOv1 Centering since it
is a normalization method which is independent of the batch size. Now although the loss strategy is
dependent on the batch size, the prototype count is highly dependent on dataset size and data modality.

For DINOv3, it was trained on a three channel modality (RGB) with a rich set of textures and variations
in intensity. Furthermore, they used a dataset of 1,700 million images from different sensors, resolutions,
camera configurations, noise profiles, locations/subjects, etc. On the other hand, sonar is a monochromatic
modality that often has repetitive textures and backgrounds with very little variation. Also, the dataset
we are using has only 1 million tiles all from the same sensor, configuration, and location. Given that,
a prototype count that is 32 times smaller than traditional DINOv3 felt appropriate. After all, the
prototype count should be small enough to force a bottleneck to motivate the model to learn, but large
enough to allow the model to express a wide-variety of rich, dense, and representative semantic features.