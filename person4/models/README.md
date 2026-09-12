PERSON 4 MODELS

`SiameseChangeModel` is a compact CPU-capable bi-temporal dense predictor. It
uses a shared three-level convolutional encoder for the before and after
images, absolute feature differences, and a decoder that returns one change
logit at every input pixel. It is suitable for supervised training on common
bi-temporal change-mask datasets. The public BIT LEVIR-CD checkpoint is stored
under `person4/checkpoints/` and is loaded by the change-detection adapter.
beeeeh
